import torch
from coref.datasets.domains.simple import (
    CapitalDomain,
    NameDomain,
    HobbyDomain,
    ObjectDomain,
    AlphabetDomain,
    FoodDomain,
    OccupationDomain,
    MaleNameDomain,
    FemaleNameDomain,
    MaleOccupationDomain,
    FemaleOccupationDomain
)
from coref.datasets.domains.common import set_prompt_id
from coref.datasets.templates.common import TrackFormatter, Substring, chat_formatter, Statement


class NameCapitalHobbyTemplate:
    default_statement_type = "plain0"
    statement_templates = {
        "plain0": " {name} lives in the capital city of {attr}.",
        "plain1": " {name} enjoys {attr}.",
        "male0": " {name} lives in the capital city of {attr}.",
        "male1": " {name} enjoys {attr}.",
        "female0": " {name} lives in the capital city of {attr}.",
        "female1": " {name} enjoys {attr}.",
        "coref0": " The {name} lives in the capital city of {attr}.",
        "coref1": " The {name} enjoys {attr}.",
        "pronoun0": " {name} lives in the capital city of {attr}.",
        "pronoun1": " {name} enjoys {attr}.",
        "rplain0": " The capital city of {attr} is inhabited by {name}.",
        "rplain1": " {attr} is enjoyed by {name}.",
    }
    corefs = ["former", "latter"]
    pronouns = ["He", "She"]

    class statement_formatters:
        def plain0(self, name, attr):
            return {"name": self.names[name], "attr": self.capitals[attr][0]}

        def plain1(self, name, attr):
            return {"name": self.names[name], "attr": self.hobbies[attr]}
        
        def male0(self, name, attr):
            return {"name": self.male_names[name], "attr": self.capitals[attr][0]}

        def male1(self, name, attr):
            return {"name": self.male_names[name], "attr": self.hobbies[attr]}
        
        def female0(self, name, attr):
            return {"name": self.female_names[name], "attr": self.capitals[attr][0]}

        def female1(self, name, attr):
            return {"name": self.female_names[name], "attr": self.hobbies[attr]}

        def rplain0(self, name, attr):
            return {"name": self.names[name], "attr": self.capitals[attr][0]}

        def rplain1(self, name, attr):
            return {"name": self.names[name], "attr": self.hobbies[attr]}

        def coref0(self, coref, attr):
            return {"name": self.corefs[coref], "attr": self.capitals[attr][0]}

        def coref1(self, coref, attr):
            return {"name": self.corefs[coref], "attr": self.hobbies[attr]}
        
        def pronoun0(self, pronoun, attr):
            return {"name": self.pronouns[pronoun], "attr": self.capitals[attr][0]}
        
        def pronoun1(self, pronoun, attr):
            return {"name": self.pronouns[pronoun], "attr": self.hobbies[attr]}

    prompt_templates = {
        "1to0": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: Which city does the person who enjoys {qn_subject} live in?\n\n"
            "Answer: The person who enjoys {qn_subject} lives in the city of"
        ),
        "0to1": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: What does the person living in {qn_subject} like?\n\n"
            "Answer: The person living in {qn_subject} likes"
        ),
        "chat_1to0": lambda chat_style, chat_history: (
            "{prompt_prefix}" + chat_formatter(
                chat_style,
                chat_history+[
                    "{context_prefix}{context}",
                    "Therefore, the person who enjoys {qn_subject} lives in the city of"
                ]
            )
        ),
        "chat_0to1": lambda chat_style, chat_history: (
            "{prompt_prefix}" + chat_formatter(
                chat_style,
                chat_history+[
                    "{context_prefix}{context}",
                    "Therefore, the person living in {qn_subject} likes"
                ]
            )
        ),
    }

    def __init__(self, tokenizer_type, split=None):
        self.names = NameDomain(tokenizer_type=tokenizer_type, split=split)
        self.capitals = CapitalDomain(tokenizer_type=tokenizer_type, split=split)
        self.hobbies = HobbyDomain(tokenizer_type=tokenizer_type, split=split)

        self.male_names = MaleNameDomain(tokenizer_type=tokenizer_type, split=split)
        self.female_names = FemaleNameDomain(tokenizer_type=tokenizer_type, split=split)

        self.tokenizer = self.names.tokenizer
        self.default_template_content = dict(
            prompt_type = "1to0",
            chat_style="llama_chat",
            prompt_prefix = "",
            context_prefix = "",
            chat_history = [],
        )
    def set_all_prompt_ids(self, prompt_id):
        return set_prompt_id(
            prompt_id,
            self.names,
            self.capitals,
            self.hobbies,
            self.male_names,
            self.female_names
        )
    def extract_template_indices(self, full_output_indices):
        return {
            "qn_subject": full_output_indices["qn_subject"][0],
            "ans_subject": full_output_indices["qn_subject"][-1],
        }

    def generate_template(
        self, prompt_id, template_content, context_content, num_answers
    ):
        """
        Returns:
            (prompt : str, template_substitutions : Dict, answer: List[int])
        """

        new_template_content = {**self.default_template_content, **{
            k: v
            for k, v in template_content.items()
            if v is not None
        }} # None values forces a fallback to default
        @lambda _: _(**new_template_content)
        def ret(query_name, prompt_type, prompt_prefix, chat_history, context_prefix, chat_style):
            with self.set_all_prompt_ids(prompt_id):
                if prompt_type in ['chat_1to0', 'chat_0to1']:
                    template = self.prompt_templates[prompt_type](chat_style=chat_style, chat_history=chat_history)
                    extra_substs = dict(
                        prompt_prefix=prompt_prefix,
                        context_prefix=context_prefix,
                    )
                else:
                    template = self.prompt_templates[prompt_type]
                    extra_substs = dict()
                if prompt_type in ["1to0", "chat_1to0"]:
                    return (
                        template,
                        dict(qn_subject=self.hobbies[query_name], **extra_substs),
                        [
                            self.capitals.encode_single_word(self.capitals[i][1])
                            for i in range(num_answers)
                        ],
                    )
                elif prompt_type in ["0to1", "chat_0to1"]:
                    return (
                        template,
                        dict(qn_subject=self.capitals[query_name][1], **extra_substs),
                        [
                            self.hobbies.encode_single_word(self.hobbies[i])
                            for i in range(num_answers)
                        ],
                    )
                else:
                    raise ValueError(f"Invalid prompt type: {prompt_type}")
        return ret

    def generate_context(self, statement_content, prompt_id, template_context):
        with self.set_all_prompt_ids(prompt_id):
            statement_type = statement_content.type or self.default_statement_type
            fields = statement_content[: len(statement_content) - 1]
            cur_ctx, ctx_idx_map = TrackFormatter().format(
                self.statement_templates[statement_type],
                **getattr(self.statement_formatters, statement_type)(self, *fields),
            )
            return cur_ctx, ctx_idx_map

    def canonize_token_maps(self, stacked_token_maps):
        """
        Extracts the location of the first token of all 'attr' and 'name'.
        Deals with parallel structure.

        Output:
        {
            'name': List[IntTensor[batch]]
            'attr': List[IntTensor[batch]]
        }
        """

        def flatten(ls):
            return [a for b in ls for a in b]

        return {
            "name": flatten(
                [
                    [statement["name"][:, 0]]
                    if isinstance(statement["name"], torch.Tensor)
                    else [k[:, 0] for k in statement["name"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr": flatten(
                [
                    [statement["attr"][:, 0]]
                    if isinstance(statement["attr"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
        }


class NameCapitalOccupationTemplate:
    default_statement_type = "plain0"
    statement_templates = {
        "plain0": " {name} lives in the capital city of {attr}.",
        "plain1": " {name} works as {attr}.",
        "male0": " {name} lives in the capital city of {attr}.",
        "male1": " {name} works as {attr}.",
        "female0": " {name} lives in the capital city of {attr}.",
        "female1": " {name} works as {attr}.",
        "coref0": " The {name} lives in the capital city of {attr}.",
        "coref1": " The {name} works as {attr}.",
        "pronoun0": " {name} lives in the capital city of {attr}.",
        "pronoun1": " {name} works as {attr}.",
        "rplain0": " The capital city of {attr} is inhabited by {name}.",
        "rplain1": " {attr} is the occupation of {name}.",
    }
    corefs = ["former", "latter"]
    pronouns = ["He", "She"]

    class statement_formatters:
        def plain0(self, name, attr):
            return {"name": self.names[name], "attr": self.capitals[attr][0]}

        def plain1(self, name, attr):
            return {"name": self.names[name], "attr": self.occupations[attr]}
        
        def male0(self, name, attr):
            return {"name": self.male_names[name], "attr": self.capitals[attr][0]}

        def male1(self, name, attr):
            return {"name": self.male_names[name], "attr": self.occupations[attr]}
        
        def female0(self, name, attr):
            return {"name": self.female_names[name], "attr": self.capitals[attr][0]}

        def female1(self, name, attr):
            return {"name": self.female_names[name], "attr": self.occupations[attr]}

        def rplain0(self, name, attr):
            return {"name": self.names[name], "attr": self.capitals[attr][0]}

        def rplain1(self, name, attr):
            return {"name": self.names[name], "attr": self.occupations[attr]}

        def coref0(self, coref, attr):
            return {"name": self.corefs[coref], "attr": self.capitals[attr][0]}

        def coref1(self, coref, attr):
            return {"name": self.corefs[coref], "attr": self.occupations[attr]}
        
        def pronoun0(self, pronoun, attr):
            return {"name": self.pronouns[pronoun], "attr": self.capitals[attr][0]}
        
        def pronoun1(self, pronoun, attr):
            return {"name": self.pronouns[pronoun], "attr": self.occupations[attr]}

    prompt_templates = {
        "1to0": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: Which city does the person who works as {qn_subject} live in?\n\n"
            "Answer: The person who enjoys {qn_subject} lives in the city of"
        ),
        "0to1": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: What does the person living in {qn_subject} work as?\n\n"
            "Answer: The person living in {qn_subject} works as"
        ),
        "chat_1to0": lambda chat_style, chat_history: (
            "{prompt_prefix}" + chat_formatter(
                chat_style,
                chat_history+[
                    "{context_prefix}{context}",
                    "Therefore, the person who works as {qn_subject} lives in the city of"
                ]
            )
        ),
        "chat_0to1": lambda chat_style, chat_history: (
            "{prompt_prefix}" + chat_formatter(
                chat_style,
                chat_history+[
                    "{context_prefix}{context}",
                    "Therefore, the person living in {qn_subject} works as"
                ]
            )
        ),
    }

    def __init__(self, tokenizer_type, split=None):
        self.names = NameDomain(tokenizer_type=tokenizer_type, split=split)
        self.capitals = CapitalDomain(tokenizer_type=tokenizer_type, split=split)
        self.occupations = OccupationDomain(tokenizer_type=tokenizer_type, split=split)

        self.male_names = MaleNameDomain(tokenizer_type=tokenizer_type, split=split)
        self.female_names = FemaleNameDomain(tokenizer_type=tokenizer_type, split=split)

        self.tokenizer = self.names.tokenizer
        self.default_template_content = dict(
            prompt_type = "1to0",
            chat_style="llama_chat",
            prompt_prefix = "",
            context_prefix = "",
            chat_history = [],
        )
    def set_all_prompt_ids(self, prompt_id):
        return set_prompt_id(
            prompt_id,
            self.names,
            self.capitals,
            self.occupations,
            self.male_names,
            self.female_names
        )
    def extract_template_indices(self, full_output_indices):
        return {
            "qn_subject": full_output_indices["qn_subject"][0],
            "ans_subject": full_output_indices["qn_subject"][-1],
        }

    def generate_template(
        self, prompt_id, template_content, context_content, num_answers
    ):
        """
        Returns:
            (prompt : str, template_substitutions : Dict, answer: List[int])
        """

        new_template_content = {**self.default_template_content, **{
            k: v
            for k, v in template_content.items()
            if v is not None
        }} # None values forces a fallback to default
        @lambda _: _(**new_template_content)
        def ret(query_name, prompt_type, prompt_prefix, chat_history, context_prefix, chat_style):
            with self.set_all_prompt_ids(prompt_id):
                if prompt_type in ['chat_1to0', 'chat_0to1']:
                    template = self.prompt_templates[prompt_type](chat_style=chat_style, chat_history=chat_history)
                    extra_substs = dict(
                        prompt_prefix=prompt_prefix,
                        context_prefix=context_prefix,
                    )
                else:
                    template = self.prompt_templates[prompt_type]
                    extra_substs = dict()
                if prompt_type in ["1to0", "chat_1to0"]:
                    return (
                        template,
                        dict(qn_subject=self.occupations[query_name], **extra_substs),
                        [
                            self.capitals.encode_single_word(self.capitals[i][1])
                            for i in range(num_answers)
                        ],
                    )
                elif prompt_type in ["0to1", "chat_0to1"]:
                    return (
                        template,
                        dict(qn_subject=self.capitals[query_name][1], **extra_substs),
                        [
                            self.occupations.encode_single_word(self.occupations[i])
                            for i in range(num_answers)
                        ],
                    )
                else:
                    raise ValueError(f"Invalid prompt type: {prompt_type}")
        return ret

    def generate_context(self, statement_content, prompt_id, template_context):
        with self.set_all_prompt_ids(prompt_id):
            statement_type = statement_content.type or self.default_statement_type
            fields = statement_content[: len(statement_content) - 1]
            cur_ctx, ctx_idx_map = TrackFormatter().format(
                self.statement_templates[statement_type],
                **getattr(self.statement_formatters, statement_type)(self, *fields),
            )
            return cur_ctx, ctx_idx_map

    def canonize_token_maps(self, stacked_token_maps):
        """
        Extracts the location of the first token of all 'attr' and 'name'.
        Deals with parallel structure.

        Output:
        {
            'name': List[IntTensor[batch]]
            'attr': List[IntTensor[batch]]
        }
        """

        def flatten(ls):
            return [a for b in ls for a in b]

        return {
            "name": flatten(
                [
                    [statement["name"][:, 0]]
                    if isinstance(statement["name"], torch.Tensor)
                    else [k[:, 0] for k in statement["name"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr": flatten(
                [
                    [statement["attr"][:, 0]]
                    if isinstance(statement["attr"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
        }
    
class GenderCapitalOccupationTemplate:
    # name is always referring to a person (either gender or occupation)
    # attr is always the location
    default_statement_type = "plain0"
    statement_templates = {
        "occupation": " The {name} works in the capital city of {attr}.",
        "capital": " The person in {attr} is {name}.",
    }
    genders = ['male', 'female']

    def get_occupation(self, occupation_id):
        # dove tail the male and female occupations
        # even occupations are male, odd are female
        if occupation_id % 2 == 0:
            return self.male_occupations[occupation_id // 2]
        elif occupation_id % 2 == 1:
            return self.female_occupations[occupation_id // 2]
        raise ValueError(f"Invalid occupation id: {occupation_id}")
    
    class statement_formatters:
        def occupation(self, name, attr):
            return {"name": self.get_occupation(name), "attr": self.capitals[attr][0]}

        def capital(self, name, attr):
            return {"name": self.genders[name], "attr": self.capitals[attr][1]}
    prompt_templates = {
        "chat": lambda chat_style, chat_history: (
            "{prompt_prefix}" + chat_formatter(
                chat_style,
                chat_history+[
                    "{context_prefix}{context}",
                    "Therefore, the gender of the {qn_subject} is"
                ]
            )
        ),
    }

    def __init__(self, tokenizer_type, split=None):
        self.capitals = CapitalDomain(tokenizer_type=tokenizer_type, split=split)
        self.female_occupations = FemaleOccupationDomain(tokenizer_type=tokenizer_type, split=split)
        self.male_occupations = MaleOccupationDomain(tokenizer_type=tokenizer_type, split=split)


        self.tokenizer = self.capitals.tokenizer
        self.default_template_content = dict(
            prompt_type = "chat",
            chat_style="llama_chat",
            prompt_prefix = "",
            context_prefix = "",
            chat_history = [],
        )
    def set_all_prompt_ids(self, prompt_id):
        return set_prompt_id(
            prompt_id,
            self.capitals,
            self.male_occupations,
            self.female_occupations,
        )
    def extract_template_indices(self, full_output_indices):
        return {
            "qn_subject": full_output_indices["qn_subject"][0],
            "ans_subject": full_output_indices["qn_subject"][-1],
        }

    def generate_template(
        self, prompt_id, template_content, context_content, num_answers
    ):
        """
        Returns:
            (prompt : str, template_substitutions : Dict, answer: List[int])
        """

        new_template_content = {**self.default_template_content, **{
            k: v
            for k, v in template_content.items()
            if v is not None
        }} # None values forces a fallback to default
        @lambda _: _(**new_template_content)
        def ret(query_name, prompt_type, prompt_prefix, chat_history, context_prefix, chat_style):
            with self.set_all_prompt_ids(prompt_id):
                template = self.prompt_templates[prompt_type](chat_style=chat_style, chat_history=chat_history)
                extra_substs = dict(
                    prompt_prefix=prompt_prefix,
                    context_prefix=context_prefix,
                )
                return (
                    template,
                    dict(qn_subject=self.get_occupation(query_name), **extra_substs),
                    [
                        self.male_occupations.encode_single_word(self.genders[i])
                        for i in range(num_answers)
                    ],
                )
        return ret

    def generate_context(self, statement_content, prompt_id, template_context):
        with self.set_all_prompt_ids(prompt_id):
            statement_type = statement_content.type or self.default_statement_type
            fields = statement_content[: len(statement_content) - 1]
            cur_ctx, ctx_idx_map = TrackFormatter().format(
                self.statement_templates[statement_type],
                **getattr(self.statement_formatters, statement_type)(self, *fields),
            )
            return cur_ctx, ctx_idx_map

    def canonize_token_maps(self, stacked_token_maps):
        """
        Extracts the location of the first token of all 'attr' and 'name'.
        Deals with parallel structure.

        Output:
        {
            'name': List[IntTensor[batch]]
            'attr': List[IntTensor[batch]]
        }
        """

        def flatten(ls):
            return [a for b in ls for a in b]

        return {
            "name": flatten(
                [
                    [statement["name"][:, 0]]
                    if isinstance(statement["name"], torch.Tensor)
                    else [k[:, 0] for k in statement["name"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr": flatten(
                [
                    [statement["attr"][:, 0]]
                    if isinstance(statement["attr"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
        }
class NameCapitalTemplate:
    default_statement_type = "plain0"
    statement_templates = {
        "plain0": " {name} lives in the capital city of {attr}.",
        "parallel": " {all_names} live in the capital cities of {all_attrs} respectively.",
    }
    corefs = ["former", "latter"]

    class statement_formatters:
        def plain0(self, name, attr):
            return {"name": self.names[name], "attr": self.capitals[attr][0]}

        def parallel(self, names, attrs):
            return dict(
                [(f"name_{i}", self.names[name]) for i, name in enumerate(names)]
                + [
                    (f"attr_{i}", self.capitals[attr][0])
                    for i, attr in enumerate(attrs)
                ]
            )

    @classmethod
    def format_parallel(cls, things):
        return ", ".join(things[:-1]) + " and " + things[-1]

    default_prompt_type = "default"
    prompt_templates = {
        "default": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: Which city does {qn_subject} live in?\n\n"
            "Answer: {qn_subject} lives in the city of"
        ),
        "barebones": ("{context}\n\n" " Therefore, {qn_subject} lives in the city of"),
        "query_attr": (
            "{context}\n\n" " Therefore, the person who lives in {qn_subject} is called"
        ),
    }

    def __init__(self, tokenizer_type, split=None):
        self.names = NameDomain(tokenizer_type=tokenizer_type, split=split)
        self.capitals = CapitalDomain(tokenizer_type=tokenizer_type, split=split)
        self.tokenizer = self.names.tokenizer

    def extract_template_indices(self, full_output_indices):
        return {
            "qn_subject": full_output_indices["qn_subject"][0],
            "ans_subject": full_output_indices["qn_subject"][-1],
        }

    def generate_template(
        self, prompt_id, template_content, context_content, num_answers
    ):
        """
        Returns:
            (prompt : str, template_substitutions : Dict, answer: List[int])
        """

        @lambda _: _(**template_content)
        def ret(query_name, prompt_type=None):
            with set_prompt_id(prompt_id, self.names, self.capitals):
                prompt_type = prompt_type or self.default_prompt_type
                if prompt_type == "query_attr":
                    return (
                        self.prompt_templates[prompt_type],
                        dict(qn_subject=self.capitals[query_name][1]),
                        [
                            self.names.encode_single_word(self.names[i])
                            for i in range(num_answers)
                        ],
                    )
                else:
                    return (
                        self.prompt_templates[prompt_type],
                        dict(qn_subject=self.names[query_name]),
                        [
                            self.capitals.encode_single_word(self.capitals[i][1])
                            for i in range(num_answers)
                        ],
                    )

        return ret

    def generate_context(self, statement_content, prompt_id, template_context):
        with set_prompt_id(prompt_id, self.names, self.capitals):
            statement_type = statement_content.type or self.default_statement_type
            fields = statement_content[: len(statement_content) - 1]
            if statement_type == "parallel":
                statement_template = self.statement_templates[statement_type].format(
                    all_names=self.format_parallel(
                        [f"{{name_{i}}}" for i in range(len(statement_content.name))]
                    ),
                    all_attrs=self.format_parallel(
                        [f"{{attr_{i}}}" for i in range(len(statement_content.attr))]
                    ),
                )
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    statement_template,
                    **getattr(self.statement_formatters, statement_type)(self, *fields),
                )
                return (
                    cur_ctx,
                    {
                        "name": [
                            ctx_idx_map[f"name_{i}"][0]
                            for i in range(len(statement_content.name))
                        ],
                        "attr": [
                            ctx_idx_map[f"attr_{i}"][0]
                            for i in range(len(statement_content.attr))
                        ],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )
            else:
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    self.statement_templates[statement_type],
                    **getattr(self.statement_formatters, statement_type)(self, *fields),
                )
                return cur_ctx, ctx_idx_map

    def canonize_token_maps(self, stacked_token_maps):
        """
        Extracts the location of the first token of all 'attr' and 'name'.
        Deals with parallel structure.

        Output:
        {
            'name': List[IntTensor[batch]]
            'attr': List[IntTensor[batch]]
        }
        """

        def flatten(ls):
            return [a for b in ls for a in b]

        # somewhat vacuous: trackformatter in default setting returns a list of location
        # so the first conditional below will never occur
        return {
            "name": flatten(
                [
                    [statement["name"][:, 0]]
                    if isinstance(statement["name"], torch.Tensor)
                    else [k[:, 0] for k in statement["name"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr": flatten(
                [
                    [statement["attr"][:, 0]]
                    if isinstance(statement["attr"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
        }


class NameHobbyTemplate:
    default_statement_type = "plain0"
    statement_templates = {
        "plain0": " {name} enjoys {attr}.",
    }
    corefs = ["former", "latter"]

    class statement_formatters:
        def plain0(self, name, attr):
            return {"name": self.names[name], "attr": self.hobbies[attr]}

    default_prompt_type = "default"
    prompt_templates = {
        "default": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: What does {qn_subject} like?\n\n"
            "Answer: {qn_subject} likes"
        )
    }

    def __init__(self, tokenizer_type, split=None):
        self.names = NameDomain(tokenizer_type=tokenizer_type, split=split)
        self.hobbies = HobbyDomain(tokenizer_type=tokenizer_type, split=split)
        self.tokenizer = self.names.tokenizer

    def extract_template_indices(self, full_output_indices):
        return {
            "qn_subject": full_output_indices["qn_subject"][0],
            "ans_subject": full_output_indices["qn_subject"][-1],
        }

    def generate_template(
        self, prompt_id, template_content, context_content, num_answers
    ):
        """
        Returns:
            (prompt : str, template_substitutions : Dict, answer: List[int])
        """

        @lambda _: _(**template_content)
        def ret(query_name, prompt_type=None):
            with set_prompt_id(prompt_id, self.names, self.hobbies):
                return (
                    self.prompt_templates[prompt_type or self.default_prompt_type],
                    dict(qn_subject=self.names[query_name]),
                    [
                        self.hobbies.encode_single_word(self.hobbies[i])
                        for i in range(num_answers)
                    ],
                )

        return ret

    def generate_context(self, statement_content, prompt_id, template_context):
        with set_prompt_id(prompt_id, self.names, self.hobbies):
            statement_type = statement_content.type or self.default_statement_type
            fields = statement_content[: len(statement_content) - 1]
            cur_ctx, ctx_idx_map = TrackFormatter().format(
                self.statement_templates[statement_type],
                **getattr(self.statement_formatters, statement_type)(self, *fields),
            )
            return cur_ctx, ctx_idx_map

    def canonize_token_maps(self, stacked_token_maps):
        """
        Extracts the location of the first token of all 'attr' and 'name'.
        Deals with parallel structure.

        Output:
        {
            'name': List[IntTensor[batch]]
            'attr': List[IntTensor[batch]]
        }
        """

        def flatten(ls):
            return [a for b in ls for a in b]

        return {
            "name": flatten(
                [
                    [statement["name"][:, 0]]
                    if isinstance(statement["name"], torch.Tensor)
                    else [k[:, 0] for k in statement["name"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr": flatten(
                [
                    [statement["attr"][:, 0]]
                    if isinstance(statement["attr"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
        }


class BoxObjectTemplate:
    default_statement_type = "reverse"
    statement_templates = {
        "reverse": " The {attr} is in Box {name}.",
    }

    class statement_formatters:
        def reverse(self, name, attr):
            return {"name": self.names[name], "attr": self.objects[attr]}

    default_prompt_type = "default"
    prompt_templates = {
        "default": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: What does Box {qn_subject} contain?\n\n"
            "Answer: Box {qn_subject} contains the"
        ),
        "barebones": ("{context} Box {qn_subject} contains the"),
    }

    def __init__(self, tokenizer_type, split=None):
        self.names = AlphabetDomain(tokenizer_type=tokenizer_type, split=split)
        self.objects = ObjectDomain(tokenizer_type=tokenizer_type, split=split)
        self.tokenizer = self.names.tokenizer

    def extract_template_indices(self, full_output_indices):
        return {
            "qn_subject": full_output_indices["qn_subject"][0],
            "ans_subject": full_output_indices["qn_subject"][-1],
        }

    def generate_template(
        self, prompt_id, template_content, context_content, num_answers
    ):
        """
        Returns:
            (prompt : str, template_substitutions : Dict, answer: List[int])
        """

        @lambda _: _(**template_content)
        def ret(query_name, prompt_type=None):
            with set_prompt_id(prompt_id, self.names, self.objects):
                return (
                    self.prompt_templates[prompt_type or self.default_prompt_type],
                    dict(qn_subject=self.names[query_name]),
                    [
                        self.objects.encode_single_word(self.objects[i])
                        for i in range(num_answers)
                    ],
                )

        return ret

    def generate_context(self, statement_content, prompt_id, template_context):
        with set_prompt_id(prompt_id, self.names, self.objects):
            statement_type = statement_content.type or self.default_statement_type
            fields = statement_content[: len(statement_content) - 1]
            cur_ctx, ctx_idx_map = TrackFormatter().format(
                self.statement_templates[statement_type],
                **getattr(self.statement_formatters, statement_type)(self, *fields),
            )
            return cur_ctx, ctx_idx_map

    def canonize_token_maps(self, stacked_token_maps):
        """
        Extracts the location of the first token of all 'attr' and 'name'.
        Deals with parallel structure.

        Output:
        {
            'name': List[IntTensor[batch]]
            'attr': List[IntTensor[batch]]
        }
        """

        def flatten(ls):
            return [a for b in ls for a in b]

        return {
            "name": flatten(
                [
                    [statement["name"][:, 0]]
                    if isinstance(statement["name"], torch.Tensor)
                    else [k[:, 0] for k in statement["name"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr": flatten(
                [
                    [statement["attr"][:, 0]]
                    if isinstance(statement["attr"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
        }


class NameCountryTemplate:
    default_statement_type = "plain0"
    statement_templates = {
        "plain0": " {name} lives in {attr}.",
        "parallel": " {all_names} live in {all_attrs} respectively.",
    }
    corefs = ["former", "latter"]

    class statement_formatters:
        def plain0(self, name, attr):
            return {"name": self.names[name], "attr": self.capitals[attr][0]}

        def parallel(self, names, attrs):
            return dict(
                [(f"name_{i}", self.names[name]) for i, name in enumerate(names)]
                + [
                    (f"attr_{i}", self.capitals[attr][0])
                    for i, attr in enumerate(attrs)
                ]
            )

    @classmethod
    def format_parallel(cls, things):
        return ", ".join(things[:-1]) + " and " + things[-1]

    prompt_templates = {
        "default": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: Which country does {qn_subject} live in?\n\n"
            "Answer: {qn_subject} lives in the country of"
        ),
        "barebones": (
            " {context}\n\n" " Therefore, {qn_subject} lives in the country of"
        ),
        "query_attr": (
            " {context}\n\n"
            " Therefore, the person who lives in {qn_subject} is called"
        ),
        "chat_name_country":  lambda chat_style, chat_history: "{prompt_prefix}" + chat_formatter(chat_style, chat_history+["{context_prefix}{context}", "Therefore, {qn_subject} lives in the country of"]),
        "chat_country_name": lambda chat_style, chat_history: "{prompt_prefix}" + chat_formatter(chat_style, chat_history+["{context_prefix}{context}", "Therefore, the person who lives in {qn_subject} is called"]),
    }

    def __init__(self, tokenizer_type, split=None):
        self.default_template_content = dict(
            prompt_type = "chat_name_country",
            chat_style="llama_chat",
            prompt_prefix = "",
            context_prefix = "",
            chat_history = [],
        )
        self.names = NameDomain(tokenizer_type=tokenizer_type, split=split)
        self.capitals = CapitalDomain(tokenizer_type=tokenizer_type, split=split)
        self.tokenizer = self.names.tokenizer

    def extract_template_indices(self, full_output_indices):
        return {
            "qn_subject": full_output_indices["qn_subject"][0],
            "ans_subject": full_output_indices["qn_subject"][-1],
        }

    def generate_template(
        self, prompt_id, template_content, context_content, num_answers
    ):
        """
        Returns:
            (prompt : str, template_substitutions : Dict, answer: List[int])
        """

        new_template_content = {**self.default_template_content, **{
            k: v
            for k, v in template_content.items()
            if v is not None
        }} # None values forces a fallback to default
        @lambda _: _(**new_template_content)
        def ret(query_name, prompt_type, prompt_prefix, chat_history, context_prefix, chat_style):
            with set_prompt_id(prompt_id, self.names, self.capitals):
                if prompt_type in ['llama_chat', 'tulu_chat']:
                    raise Exception(
                        "Use chat_style instead to specify chat type. "
                        "prompt_type should be either chat_name_country or chat_country_name"
                    )
                if 'chat' in prompt_type:
                    template = self.prompt_templates[prompt_type](chat_style=chat_style, chat_history=chat_history)
                    prefixes = dict(
                        prompt_prefix=prompt_prefix,
                        context_prefix=context_prefix,
                    )
                else:
                    template = self.prompt_templates[prompt_type]
                    prefixes = {}
                if prompt_type == "query_attr" or prompt_type == "chat_country_name":
                    return (
                        template,
                        dict(**prefixes, qn_subject=self.capitals[query_name][0]),
                        [
                            self.names.encode_single_word(self.names[i])
                            for i in range(num_answers)
                        ],
                    )
                else:
                    return (
                        template,
                        dict(**prefixes, qn_subject=self.names[query_name]),
                        [
                            self.capitals.encode_single_word(self.capitals[i][0])
                            for i in range(num_answers)
                        ],
                    )

        return ret

    def generate_context(self, statement_content, prompt_id, template_context):
        with set_prompt_id(prompt_id, self.names, self.capitals):
            statement_type = statement_content.type or self.default_statement_type
            fields = statement_content[: len(statement_content) - 1]
            if statement_type == "parallel":
                statement_template = self.statement_templates[statement_type].format(
                    all_names=self.format_parallel(
                        [f"{{name_{i}}}" for i in range(len(statement_content.name))]
                    ),
                    all_attrs=self.format_parallel(
                        [f"{{attr_{i}}}" for i in range(len(statement_content.attr))]
                    ),
                )
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    statement_template,
                    **getattr(self.statement_formatters, statement_type)(self, *fields),
                )
                return (
                    cur_ctx,
                    {
                        "name": [
                            ctx_idx_map[f"name_{i}"][0]
                            for i in range(len(statement_content.name))
                        ],
                        "attr": [
                            ctx_idx_map[f"attr_{i}"][0]
                            for i in range(len(statement_content.attr))
                        ],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )
            else:
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    self.statement_templates[statement_type],
                    **getattr(self.statement_formatters, statement_type)(self, *fields),
                )
                return cur_ctx, ctx_idx_map

    def get_standard_context(self, num_entities):
        return [
            Statement(x, x, None)
            for x in range(num_entities)
        ]
    def get_predicates(self, context_content, prompt_id):
        true_predicates = []
        with set_prompt_id(prompt_id, self.names, self.capitals):
            for statement in context_content:
                true_predicates.append((
                    self.names[statement.name],
                    self.capitals[statement.attr],
                    self.capitals.type
                ))
        return true_predicates

    def canonize_token_maps(self, stacked_token_maps):
        """
        Extracts the location of the first token of all 'attr' and 'name'.
        Deals with parallel structure.

        Output:
        {
            'name': List[IntTensor[batch]]
            'attr': List[IntTensor[batch]]
        }
        """

        def flatten(ls):
            return [a for b in ls for a in b]

        # somewhat vacuous: trackformatter in default setting returns a list of location
        # so the first conditional below will never occur
        return {
            "name": flatten(
                [
                    [statement["name"][:, 0]]
                    if isinstance(statement["name"], torch.Tensor)
                    else [k[:, 0] for k in statement["name"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr": flatten(
                [
                    [statement["attr"][:, 0]]
                    if isinstance(statement["attr"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
        }



class NameFoodTemplate:
    default_statement_type = "plain0"
    statement_templates = {
        "plain0": " {name} likes to eat {attr}.",
        "parallel": " {all_names} like to eat {all_attrs} respectively.",
    }
    corefs = ["former", "latter"]

    class statement_formatters:
        def plain0(self, name, attr):
            return {"name": self.names[name], "attr": self.foods[attr]}

        def parallel(self, names, attrs):
            return dict(
                [(f"name_{i}", self.names[name]) for i, name in enumerate(names)]
                + [
                    (f"attr_{i}", self.foods[attr])
                    for i, attr in enumerate(attrs)
                ]
            )

    @classmethod
    def format_parallel(cls, things):
        return ", ".join(things[:-1]) + " and " + things[-1]

    default_prompt_type = "default"
    prompt_templates = {
        "default": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: What does {qn_subject} like to eat?\n\n"
            "Answer: {qn_subject} likes to eat"
        ),
        "barebones": (
            " {context}\n\n" " Therefore, {qn_subject} likes to eat"
        ),
        "query_attr": (
            " {context}\n\n"
            " Therefore, the person who likes to eat {qn_subject} is called"
        ),
        "chat_name_food":  lambda chat_style, chat_history: "{prompt_prefix}" + chat_formatter(chat_style, chat_history+["{context_prefix}{context}", "Therefore, {qn_subject} likes to eat"]),
        "chat_food_name": lambda chat_style, chat_history: "{prompt_prefix}" + chat_formatter(chat_style, chat_history+["{context_prefix}{context}", "Therefore, the person who likes to eat {qn_subject} is called"]),
    }

    def __init__(self, tokenizer_type, split=None):
        self.default_template_content = dict(
            prompt_type = "chat_name_food",
            chat_style="llama_chat",
            prompt_prefix = "",
            context_prefix = "",
            chat_history = [],
        )
        self.names = NameDomain(tokenizer_type=tokenizer_type, split=split)
        self.foods = FoodDomain(tokenizer_type=tokenizer_type, split=split)
        self.tokenizer = self.names.tokenizer

    def extract_template_indices(self, full_output_indices):
        return {
            "qn_subject": full_output_indices["qn_subject"][0],
            "ans_subject": full_output_indices["qn_subject"][-1],
        }

    def generate_template(
        self, prompt_id, template_content, context_content, num_answers
    ):
        """
        Returns:
            (prompt : str, template_substitutions : Dict, answer: List[int])
        """

        new_template_content = {**self.default_template_content, **{
            k: v
            for k, v in template_content.items()
            if v is not None
        }} # None values forces a fallback to default
        @lambda _: _(**new_template_content)
        def ret(query_name, prompt_type, prompt_prefix, chat_history, context_prefix, chat_style):
            with set_prompt_id(prompt_id, self.names, self.foods):
                if 'chat' in prompt_type:
                    template = self.prompt_templates[prompt_type](chat_style=chat_style, chat_history=chat_history)
                    prefixes = dict(
                        prompt_prefix=prompt_prefix,
                        context_prefix=context_prefix,
                    )
                else:
                    template = self.prompt_templates[prompt_type]
                    prefixes = {}
                if prompt_type == "query_attr" or prompt_type == "chat_food_name":
                    return (
                        template,
                        dict(**prefixes, qn_subject=self.foods[query_name]),
                        [
                            self.names.encode_single_word(self.names[i])
                            for i in range(num_answers)
                        ],
                    )
                else:
                    return (
                        template,
                        dict(**prefixes, qn_subject=self.names[query_name]),
                        [
                            self.foods.encode_single_word(self.foods[i], suppress_error=True)
                            for i in range(num_answers)
                        ],
                    )

        return ret

    def generate_context(self, statement_content, prompt_id, template_context):
        with set_prompt_id(prompt_id, self.names, self.foods):
            statement_type = statement_content.type or self.default_statement_type
            fields = statement_content[: len(statement_content) - 1]
            if statement_type == "parallel":
                statement_template = self.statement_templates[statement_type].format(
                    all_names=self.format_parallel(
                        [f"{{name_{i}}}" for i in range(len(statement_content.name))]
                    ),
                    all_attrs=self.format_parallel(
                        [f"{{attr_{i}}}" for i in range(len(statement_content.attr))]
                    ),
                )
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    statement_template,
                    **getattr(self.statement_formatters, statement_type)(self, *fields),
                )
                return (
                    cur_ctx,
                    {
                        "name": [
                            ctx_idx_map[f"name_{i}"][0]
                            for i in range(len(statement_content.name))
                        ],
                        "attr": [
                            ctx_idx_map[f"attr_{i}"][0]
                            for i in range(len(statement_content.attr))
                        ],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )
            else:
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    self.statement_templates[statement_type],
                    **getattr(self.statement_formatters, statement_type)(self, *fields),
                )
                return cur_ctx, ctx_idx_map

    def get_standard_context(self, num_entities):
        return [
            Statement(x, x, None)
            for x in range(num_entities)
        ]

    def get_predicates(self, context_content, prompt_id):
        true_predicates = []
        with set_prompt_id(prompt_id, self.names, self.foods):
            for statement in context_content:
                true_predicates.append((
                    self.names[statement.name],
                    self.foods[statement.attr],
                    self.foods.type
                ))
        return true_predicates
    def canonize_token_maps(self, stacked_token_maps):
        """
        Extracts the location of the first token of all 'attr' and 'name'.
        Deals with parallel structure.

        Output:
        {
            'name': List[IntTensor[batch]]
            'attr': List[IntTensor[batch]]
        }
        """

        def flatten(ls):
            return [a for b in ls for a in b]

        # somewhat vacuous: trackformatter in default setting returns a list of location
        # so the first conditional below will never occur
        return {
            "name": flatten(
                [
                    [statement["name"][:, 0]]
                    if isinstance(statement["name"], torch.Tensor)
                    else [k[:, 0] for k in statement["name"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr": flatten(
                [
                    [statement["attr"][:, 0]]
                    if isinstance(statement["attr"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
        }
    

class NameOccupationTemplate:
    default_statement_type = "plain0"
    statement_templates = {
        "plain0": " {name} works as {attr}.",
        "parallel": " {all_names} work as {all_attrs} respectively.",
    }
    corefs = ["former", "latter"]

    class statement_formatters:
        def plain0(self, name, attr):
            return {"name": self.names[name], "attr": self.occupations[attr]}

        def parallel(self, names, attrs):
            return dict(
                [(f"name_{i}", self.names[name]) for i, name in enumerate(names)]
                + [
                    (f"attr_{i}", self.occupations[attr])
                    for i, attr in enumerate(attrs)
                ]
            )

    @classmethod
    def format_parallel(cls, things):
        return ", ".join(things[:-1]) + " and " + things[-1]

    default_prompt_type = "default"
    prompt_templates = {
        "default": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: What is the occupation of {qn_subject}?\n\n"
            "Answer: The occupation of {qn_subject} is"
        ),
        "barebones": (
            " {context}\n\n" " Therefore, the occupation of {qn_subject} is"
        ),
        "query_attr": (
            " {context}\n\n"
            " Therefore, the person who works as {qn_subject} is called"
        ),
        "chat_name_occupation":  lambda chat_style, chat_history: "{prompt_prefix}" + chat_formatter(chat_style, chat_history+["{context_prefix}{context}", "Therefore, the occupation of {qn_subject} is"]),
        "chat_occupation_name": lambda chat_style, chat_history: "{prompt_prefix}" + chat_formatter(chat_style, chat_history+["{context_prefix}{context}", "Therefore, the person who works as a {qn_subject} is called"]),
    }

    def __init__(self, tokenizer_type, split=None):
        self.default_template_content = dict(
            prompt_type = "chat_name_occupation",
            chat_style="llama_chat",
            prompt_prefix = "",
            context_prefix = "",
            chat_history = [],
        )
        self.names = NameDomain(tokenizer_type=tokenizer_type, split=split)
        self.occupations = OccupationDomain(tokenizer_type=tokenizer_type, split=split)
        self.tokenizer = self.names.tokenizer

    def extract_template_indices(self, full_output_indices):
        return {
            "qn_subject": full_output_indices["qn_subject"][0],
            "ans_subject": full_output_indices["qn_subject"][-1],
        }

    def generate_template(
        self, prompt_id, template_content, context_content, num_answers
    ):
        """
        Returns:
            (prompt : str, template_substitutions : Dict, answer: List[int])
        """

        new_template_content = {**self.default_template_content, **{
            k: v
            for k, v in template_content.items()
            if v is not None
        }} # None values forces a fallback to default
        @lambda _: _(**new_template_content)
        def ret(query_name, prompt_type, prompt_prefix, chat_history, context_prefix, chat_style):
            with set_prompt_id(prompt_id, self.names, self.occupations):
                if 'chat' in prompt_type:
                    template = self.prompt_templates[prompt_type](chat_style=chat_style, chat_history=chat_history)
                    prefixes = dict(
                        prompt_prefix=prompt_prefix,
                        context_prefix=context_prefix,
                    )
                else:
                    template = self.prompt_templates[prompt_type]
                    prefixes = {}
                if prompt_type == "query_attr" or prompt_type == "chat_occupation_name":
                    return (
                        template,
                        dict(**prefixes, qn_subject=self.occupations[query_name]),
                        [
                            self.names.encode_single_word(self.names[i])
                            for i in range(num_answers)
                        ],
                    )
                else:
                    return (
                        template,
                        dict(**prefixes, qn_subject=self.names[query_name]),
                        [
                            self.occupations.encode_single_word(self.occupations[i],)
                            for i in range(num_answers)
                        ],
                    )

        return ret

    def generate_context(self, statement_content, prompt_id, template_context):
        with set_prompt_id(prompt_id, self.names, self.occupations):
            statement_type = statement_content.type or self.default_statement_type
            fields = statement_content[: len(statement_content) - 1]
            if statement_type == "parallel":
                statement_template = self.statement_templates[statement_type].format(
                    all_names=self.format_parallel(
                        [f"{{name_{i}}}" for i in range(len(statement_content.name))]
                    ),
                    all_attrs=self.format_parallel(
                        [f"{{attr_{i}}}" for i in range(len(statement_content.attr))]
                    ),
                )
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    statement_template,
                    **getattr(self.statement_formatters, statement_type)(self, *fields),
                )
                return (
                    cur_ctx,
                    {
                        "name": [
                            ctx_idx_map[f"name_{i}"][0]
                            for i in range(len(statement_content.name))
                        ],
                        "attr": [
                            ctx_idx_map[f"attr_{i}"][0]
                            for i in range(len(statement_content.attr))
                        ],
                        "sentence": Substring(0, len(cur_ctx)),
                    },
                )
            else:
                cur_ctx, ctx_idx_map = TrackFormatter().format(
                    self.statement_templates[statement_type],
                    **getattr(self.statement_formatters, statement_type)(self, *fields),
                )
                return cur_ctx, ctx_idx_map

    def get_standard_context(self, num_entities):
        return [
            Statement(x, x, None)
            for x in range(num_entities)
        ]

    def get_predicates(self, context_content, prompt_id):
        true_predicates = []
        with set_prompt_id(prompt_id, self.names, self.occupations):
            for statement in context_content:
                true_predicates.append((
                    self.names[statement.name],
                    self.occupations[statement.attr],
                    self.occupations.type
                ))
        return true_predicates
    def canonize_token_maps(self, stacked_token_maps):
        """
        Extracts the location of the first token of all 'attr' and 'name'.
        Deals with parallel structure.

        Output:
        {
            'name': List[IntTensor[batch]]
            'attr': List[IntTensor[batch]]
        }
        """

        def flatten(ls):
            return [a for b in ls for a in b]

        # somewhat vacuous: trackformatter in default setting returns a list of location
        # so the first conditional below will never occur
        return {
            "name": flatten(
                [
                    [statement["name"][:, 0]]
                    if isinstance(statement["name"], torch.Tensor)
                    else [k[:, 0] for k in statement["name"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr": flatten(
                [
                    [statement["attr"][:, 0]]
                    if isinstance(statement["attr"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
        }