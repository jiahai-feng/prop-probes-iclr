import torch
from coref.datasets.domains.simple import CapitalDomain, NameDomain, HobbyDomain
from coref.datasets.domains.common import set_prompt_id
from coref.datasets.templates.common import TrackFormatter


class TripletNameCapitalHobbyTemplate:
    default_statement_type = "plain0"
    statement_templates = {
        "plain0": " {attr0} from {attr1} likes {attr2}.",
    }
    corefs = ["former", "latter"]

    class statement_formatters:
        def plain0(self, attr0, attr1, attr2):
            return {
                "attr0": self.names[attr0],
                "attr1": self.capitals[attr1][0],
                "attr2": self.hobbies[attr2],
            }

    default_prompt_type = "attr0"
    prompt_templates = {
        "attr0": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: Who from {attr1} likes {attr2}\n\n"
            "Answer: The person from {attr1} who likes {attr2} is"
        ),
        "attr1": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: Which country is {attr0} who likes {attr2} from?\n\n"
            "Answer: {attr0} who likes {attr2} is from"
        ),
        "attr2": (
            "Answer the question based on the context below. Keep the answer short.\n\n"
            "Context:{context}\n\n"
            "Question: What does {attr0} from {attr1} like?\n\n"
            "Answer: {attr0} from {attr1} likes"
        ),
    }

    class prompt_formatters:
        def attr0(self, attr1, attr2):
            return {"attr1": self.capitals[attr1][0], "attr2": self.hobbies[attr2]}

        def attr1(self, attr0, attr2):
            return {"attr0": self.names[attr0], "attr2": self.hobbies[attr2]}

        def attr2(self, attr0, attr1):
            return {"attr0": self.names[attr0], "attr1": self.capitals[attr1][0]}

    def __init__(self, tokenizer_type, split=None):
        self.names = NameDomain(tokenizer_type=tokenizer_type, split=split)
        self.capitals = CapitalDomain(tokenizer_type=tokenizer_type, split=split)
        self.hobbies = HobbyDomain(tokenizer_type=tokenizer_type, split=split)
        self.tokenizer = self.names.tokenizer

    def extract_template_indices(self, full_output_indices):
        return {}

    def generate_template(
        self, prompt_id, template_content, context_content, num_answers
    ):
        """
        Returns:
            (prompt : str, template_substitutions : Dict, answer: List[int])
        """

        @lambda _: _(**template_content)
        def ret(attrA, attrB, prompt_type=None):
            with set_prompt_id(prompt_id, self.names, self.capitals, self.hobbies):
                prompt_type = prompt_type or self.default_prompt_type
                answer_attrs = {
                    "attr0": lambda i: self.names.encode_single_word(self.names[i]),
                    "attr1": lambda i: self.capitals.encode_single_word(
                        self.capitals[i][0]
                    ),
                    "attr2": lambda i: self.hobbies.encode_single_word(self.hobbies[i]),
                }
                return (
                    self.prompt_templates[prompt_type],
                    getattr(self.prompt_formatters, prompt_type)(self, attrA, attrB),
                    [answer_attrs[prompt_type](i) for i in range(num_answers)],
                )

        return ret

    def generate_context(self, statement_content, prompt_id, template_context):
        with set_prompt_id(prompt_id, self.names, self.capitals, self.hobbies):
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
            'attr0': List[IntTensor[batch]]
            'attr1': List[IntTensor[batch]]
            'attr2': List[IntTensor[batch]]
        }
        """

        def flatten(ls):
            return [a for b in ls for a in b]

        return {
            "attr0": flatten(
                [
                    [statement["attr0"][:, 0]]
                    if isinstance(statement["attr0"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr0"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr1": flatten(
                [
                    [statement["attr1"][:, 0]]
                    if isinstance(statement["attr1"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr1"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
            "attr2": flatten(
                [
                    [statement["attr2"][:, 0]]
                    if isinstance(statement["attr2"], torch.Tensor)
                    else [k[:, 0] for k in statement["attr2"]]
                    for statement in stacked_token_maps["context"]
                ]
            ),
        }
