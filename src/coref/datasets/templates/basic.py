from coref.datasets.templates.common import TrackFormatter, chat_formatter, drop_none
from coref.models import get_tokenizer

class BasicPrefixTemplate:
    prompt_templates = dict(
        llama_chat=lambda: chat_formatter("llama_chat", [
            "{prefix} {context}", "{query}"
        ]),
        tulu_chat=lambda: chat_formatter("tulu_chat", [
            "{prefix} {context}", "{query}"
        ])
    )
    def __init__(self, tokenizer_type, split=None):
        self.default_template_content = dict(
            prompt_style="llama_chat",
        )
        self.tokenizer_type = tokenizer_type
        self.tokenizer = get_tokenizer(tokenizer_type)

    def generate_context(self, statement_content, prompt_id, template_context):
        raise ValueError("Basic templates don't have contexts")
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
        
        new_template_content = {**self.default_template_content, **{
            k: v
            for k, v in template_content.items()
            if v is not None
        }} # None values forces a fallback to default
        @lambda _: _(**new_template_content)
        def ret(prompt_style, prefix, context, query):
            return (
                self.prompt_templates[prompt_style](),
                dict(
                    prefix=prefix,
                    context=context,
                    query=query,
                ),
                []
            )
        return ret
    def extract_template_indices(self, full_output_indices):
        '''
        Extracts the indices of any field we want to retain
        Typically used to retain template substitutions
        Args:
            full_output_indices : Dict[str, List[Substring]]
        Returns:
            Dict[str, Substring]
        '''
        return {
            "prefix": full_output_indices["prefix"][0],
            "context": full_output_indices["context"][0],
            "query": full_output_indices["query"][0],
        }
    

class WinobiasTemplate:
    prompt_templates = dict(
        llama_chat=lambda : chat_formatter("llama_chat", [
            "{prefix} {context}", "{query}"
        ]),
        tulu_chat=lambda : chat_formatter("tulu_chat", [
            "{prefix} {context}", "{query}"
        ])
    )
    def __init__(self, tokenizer_type, split=None):
        self.default_template_content = dict(
            prompt_style="llama_chat",
        )
        self.tokenizer_type = tokenizer_type
        self.tokenizer = get_tokenizer(tokenizer_type)

    def generate_context(self, statement_content, prompt_id, template_context):
                    
        new_template_content = {
            **self.default_template_content, 
            **drop_none(template_context),  # None values forces a fallback to default
            **drop_none(statement_content)
        }
        @lambda _: _(**new_template_content)
        def ret(template, subject, distractor, pronoun, **kwargs):
            cur_ctx, ctx_idx_map = TrackFormatter().format(
                template,
                subject=subject,
                distractor=distractor,
                pronoun=pronoun,
            )

            return (
                cur_ctx,
                {
                    key: ctx_idx_map[key][0]
                    for key in ['subject', 'distractor', 'pronoun']
                },
            )
        return ret
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
        
        new_template_content = {**self.default_template_content, **{
            k: v
            for k, v in template_content.items()
            if v is not None
        }} # None values forces a fallback to default
        @lambda _: _(**new_template_content)
        def ret(prompt_style, prefix, query):
            return (
                self.prompt_templates[prompt_style](),
                dict(
                    query=query,
                    prefix=prefix,
                ),
                []
            )
        return ret
    def extract_template_indices(self, full_output_indices):
        '''
        Extracts the indices of any field we want to retain
        Typically used to retain template substitutions
        Args:
            full_output_indices : Dict[str, List[Substring]]
        Returns:
            Dict[str, Substring]
        '''
        return {
            "prefix": full_output_indices["prefix"][0],
            "query": full_output_indices["query"][0],
        }