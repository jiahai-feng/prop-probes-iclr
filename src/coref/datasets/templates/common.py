from dataclasses import dataclass
from typing import List, Union
from string import Formatter
from collections import defaultdict, namedtuple
import itertools

import numpy as np
import torch

    

Statement = namedtuple("Statement", ["name", "attr", "type"], defaults=[None])
TStatement = namedtuple("Statement", ["attr0", "attr1", "attr2", "type"], defaults=[None])
VarStatement = namedtuple("Statement", ["fields", "type"], defaults=[None])
class TrackFormatter(Formatter):
    def format(self, format_string, **kwargs):
        """
        Only accepts keyword arguments.

        Returns:
            formatted string : str
            locations : Dict[str, List[Tuple[int, int]]] - denoting start and end positions (inclusive, exclusive)
        """
        locations = defaultdict(list)
        result = []
        run_length = 0
        for literal_text, field_name, format_spec, conversion in self.parse(
            format_string
        ):
            # output the literal text
            if literal_text:
                result.append(literal_text)
                run_length += len(literal_text)

            if field_name is not None:
                # given the field_name, find the object it references
                #  and the argument it came from
                obj, arg_used = self.get_field(field_name, [], kwargs)

                # do any conversion on the resulting object
                obj = self.convert_field(obj, conversion)

                # format the object and append to the result
                final_str = self.format_field(obj, format_spec)
                locations[arg_used].append(
                    Substring(run_length, run_length + len(final_str))
                )
                result.append(final_str)
                run_length += len(final_str)

        return "".join(result), locations


from dataclasses import dataclass


@dataclass(frozen=True)
class Substring:
    start: int
    end: int

    def __iter__(self):
        return iter((self.start, self.end))

    def __getitem__(self, key):
        if key == 0:
            return self.start
        else:
            return self.end

    def to_slice(self):
        return slice(self.start, self.end)    
    def to_list(self):
        return [self.start, self.end]

    def __add__(self, num):
        return Substring(self.start + num, self.end + num)


import bisect
from functools import partial

def recursify_accessor(func, dtype=Substring, pred=None):
    '''
    Returns a function that recursively traverses a data structure,
    until we reach a certain type or when the predicate is satisfied,
    at which case we apply the function, providing it with the object
    that we reached, as well as an accessor function that, when applied
    to the same data structure, will return the object that we reached.

    This is useful for traversing multiple data structures of the same 
    keys, e.g. zipping together hierarchical data structures.
    '''
    if pred is None:
        pred = lambda x: isinstance(x, dtype)

    def wrapper(indices, accessor, *args, **kwargs):
        if pred(indices):
            return func(indices, accessor, *args, **kwargs)
        elif isinstance(indices, dict):
            return {
                key: wrapper(value, lambda x: accessor(x)[key],*args, **kwargs) for key, value in indices.items()
            }
        elif isinstance(indices, list):
            return [wrapper(value, lambda x: accessor(x)[i], *args, **kwargs) for i, value in enumerate(indices)]
        else:
            raise Exception(f"Unexpected type {type(indices)}")

    return partial(wrapper, accessor=lambda x: x)

def recursify(func, dtype=Substring, pred=None):
    if pred is None:
        pred = lambda x: isinstance(x, dtype)

    def wrapper(indices, *args, **kwargs):
        if pred(indices):
            return func(indices, *args, **kwargs)
        elif isinstance(indices, dict):
            return {
                key: wrapper(value, *args, **kwargs) for key, value in indices.items()
            }
        elif isinstance(indices, list):
            return [wrapper(value, *args, **kwargs) for value in indices]
        else:
            raise Exception(f"Unexpected type {type(indices)}")

    return wrapper


@recursify
def recursive_align_tokens(indices, offset_mapping):
    # it's unclear what conventions offset_mapping uses
    # but we can say for sure that starting indices are inclusive
    # but my indices are inclusive, exclusive
    # When indices is empty string, we return empty indices (i.e. start = end)
    start, end = indices
    # bisect_right gives the first token that starts strictly after the search
    tok_start = bisect.bisect_right([x for x, _ in offset_mapping], start) - 1
    tok_end = bisect.bisect_right([x for x, _ in offset_mapping], end - 1) - 1
    if start == end:
        return Substring(tok_start, tok_start)
    else:
        return Substring(tok_start, tok_end + 1)


@recursify
def recursive_add_offset(indices, offset):
    return indices + offset


def rotate(func, dtype=Substring, pred=None):
    """
    Rotates "List ... x" into "... List x", and calls func on List x

    Returns:
        ... func(List x)
    """
    if pred is None:
        pred = lambda x: isinstance(x, dtype)

    def wrapper(indices, *args, **kwargs):
        assert isinstance(indices, list)
        child = indices[0]
        if pred(child):
            return func(indices, *args, **kwargs)
        elif isinstance(child, dict):
            return {
                key: wrapper([sib[key] for sib in indices], *args, **kwargs)
                for key in child.keys()
            }

        elif isinstance(child, list):
            return [
                wrapper([sib[key] for sib in indices], *args, **kwargs)
                for key in range(len(child))
            ]
        else:
            raise Exception(f"Unexpected type {type(child)}")

    return wrapper


def tokenize_prompt(*, template, prompt, indices, answers, **kwargs):
    tokenized = template.tokenizer(prompt, return_offsets_mapping=True)
    aligned_tokens = recursive_align_tokens(indices, tokenized["offset_mapping"])
    return tokenized["input_ids"], aligned_tokens, answers
    


def stack_tokens(template, tokens_list):
    longest_length = max(len(s) for s in tokens_list)
    assert type(template.tokenizer.pad_token_type_id) == int
    padded = torch.tensor(
        [
            s + [template.tokenizer.pad_token_type_id] * (longest_length - len(s))
            for s in tokens_list
        ]
    )
    return padded

def chat_formatter(style, messages):
    '''
    Args:
        style: Literal["llama_chat", "tulu_chat", "sep"]
        messages: List[str]
    
    References:
    - Tulu: https://github.com/allenai/open-instruct/blob/main/open_instruct/finetune.py
            encode_with_messages_format
    - Llama: https://github.com/meta-llama/llama/blob/main/llama/generation.py
            chat_completion  
    '''
    assert style in {"llama_chat", "tulu_chat", 'sep'}
    templates = {
        'tulu_chat': "<|user|>\n{user}\n<|assistant|>\n{asst}",
        'llama_chat': "[INST] {user} [/INST] {asst}",
        'sep': "{user}<|sep|>{asst}"
    }
    query_templates = {
        'tulu_chat': "<|user|>\n{user}\n<|assistant|>\n",
        'llama_chat': "[INST] {user} [/INST]",
        'sep': "{user}<|sep|>"
    }
    joins = {
        'tulu_chat': "</s>\n",
        'llama_chat': "</s><s>",
        'sep': '<|sep|>'
    }
    formatted_messages = []
    for user, asst in zip(messages[::2], messages[1::2]):
        formatted_messages.append(templates[style].format(user=user, asst=asst))
    if len(messages) % 2 != 0:
        formatted_messages.append(query_templates[style].format(user=messages[-1]))
    return joins[style].join(formatted_messages)



from contextlib import contextmanager

@contextmanager
def set_defaults(template, new_defaults):
    for key in new_defaults:
        assert key in template.default_template_content, f"Unknown default attribute {key} in {new_defaults}"
    previous = {key: template.default_template_content[key] for key in new_defaults}
    for key, value in new_defaults.items():
        template.default_template_content[key] = value
    try:
        yield template
    finally:
        for key, value in previous.items():
            template.default_template_content[key] = value


'''
Specification for template
Note: at present, templates don't actually inherit from this class
This is just a specification imposed by api.generate_prompts
'''
class AbstractTemplate:
    def generate_context(self, statement_content, prompt_id, template_context):
        '''
        Return text, indices for every statement in the context
        Args:
            statement_content : Statement
            prompt_id : int
            template_context : Dict[str, Any]
        Returns
            text : str
            indices : Recursifiable[Substring]
        '''
        raise NotImplementedError
    def generate_template(self, prompt_id, template_content, context_content, num_answers):
        '''
        Returns the template that we're instantiating.
        The template_string must contain the following fields:
        - context
        - every key in template_substitutions

        Args:
            prompt_id : int
            template_content : Dict[str, Any]
            context_content : List[Statement]
            num_answers : int
        Returns:
            template_string : str
            template_substitutions : Dict[str, str]
            answers : List[int]
        '''
        raise NotImplementedError
    def extract_template_indices(self, full_output_indices):
        '''
        Extracts the indices of any field we want to retain
        Typically used to retain template substitutions
        Args:
            full_output_indices : Dict[str, List[Substring]]
        Returns:
            Dict[str, Substring]
        '''
        raise NotImplementedError

def drop_none(d):
    return {k: v for k, v in d.items() if v is not None}

# Unclear if useful
# @dataclass
# class ContextStatement:
#     fields: List[int]
#     type: Union[str, None]
#     @classmethod
#     def from_namedtuple(cls, tup):
#         if 'type' in tup._fields:
#             statement_type = tup.type
#         else:
#             statement_type = None
#         return cls(
#             fields=[tup[i] for i in range(len(tup._fields) - 1)],
#             type=statement_type
#         )

