import itertools
from coref.utils import prepend
from coref.datasets.templates.common import (
    TrackFormatter,
    Substring,
    recursive_add_offset,
    tokenize_prompt,
)

import coref.datasets.templates.simple as ts
import coref.datasets.templates.fixed as tf


template_dict = dict(
    NameCapitalHobbyTemplate=ts.NameCapitalHobbyTemplate,
    NameCapitalTemplate=ts.NameCapitalTemplate,
    NameHobbyTemplate=ts.NameHobbyTemplate,
    BoxObjectTemplate=ts.BoxObjectTemplate,
    NameCountryTemplate=ts.NameCountryTemplate,
    NameCountryFoodFixedTemplate=tf.NameCountryFoodFixedTemplate,
    NameCountryFoodOccupationFixedTemplate=tf.NameCountryFoodOccupationFixedTemplate,
    NameCountryOccupationFixedTemplate=tf.NameCountryOccupationFixedTemplate,
    NameCountryVaryFixedTemplate=tf.NameCountryVaryFixedTemplate
)


def get_template(template_name):
    return template_dict[template_name]


def generate_prompts(
    *,
    template,
    template_content,
    context_content,
    prompt_id,
    num_answers,
    **kwargs,
):
    """
    Args:
        context_content: List[Statement]
    """

    template_string, template_substitutions, answers = template.generate_template(
        template_content=template_content,
        prompt_id=prompt_id,
        context_content=context_content,
        num_answers=num_answers,
    )
    if context_content:
        context_text = []
        context_indices = []
        for statement_content in context_content:
            text, indices = template.generate_context(
                statement_content=statement_content,
                prompt_id=prompt_id,
                template_context=template_content,
            )
            context_text.append(text)
            context_indices.append(indices)
        joined_context_text = "".join(context_text)
        context_substitutions = dict(context=joined_context_text)
    else:
        context_substitutions = {}
    full_output, full_output_indices = TrackFormatter().format(
        template_string, **context_substitutions, **template_substitutions
    )

    if context_content:
        context_start_pos = full_output_indices["context"][0].start
        context_acc = itertools.accumulate(
            [context_start_pos] + [len(x) for x in context_text]
        )
        context_indices = {
            "context_section": full_output_indices["context"][0],
            "context": [
                recursive_add_offset(ctx_idx_map, offset)
                for offset, ctx_idx_map in zip(context_acc, context_indices)
            ]
        }
    else:
        context_indices = {}


    full_output_indices = {
        "prompt": Substring(0, len(full_output)),
        **context_indices,
        **template.extract_template_indices(full_output_indices),
    }
    return dict(prompt=full_output, indices=full_output_indices, answers=answers)


generate_tokenized_prompts = prepend(generate_prompts)(tokenize_prompt)
