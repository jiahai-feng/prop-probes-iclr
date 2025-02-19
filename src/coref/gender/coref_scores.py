import torch
import coref.datasets.templates.basic as tb
import coref.datasets.api as ta

import coref.probes.activation_localization as act_loc

import einops

import coref.probes.lookup_probe as lkp
import coref.form_processors
import coref.gender.eval as cge



from collections.abc import Sequence

class SliceRange(Sequence):
    def __init__(self, start, stop, batch_size):
        self.start = start
        self.stop = stop
        self.batch_size = batch_size
    def __getitem__(self, index):
        start = self.start + index * self.batch_size
        stop = min(self.stop, start + self.batch_size)
        if start >= self.stop:
            raise IndexError
        return slice(start, stop)
    def __len__(self) -> int:
        return (self.stop - self.start + self.batch_size - 1) // self.batch_size
    
def pad_right(input_tokens, pad_token_id=1):
    max_length = max(len(s) for s in input_tokens)
    padded = torch.tensor(
        [
            s + [pad_token_id] * (max_length - len(s))
            for s in input_tokens
        ]
    )
    pad_lengths = torch.tensor([max_length - len(s) for s in input_tokens])
    return padded, pad_lengths


def cache_to_norm_acts(model, cache):
    acts = einops.rearrange(
        torch.stack([cache[f'blocks.{layer}.hook_resid_pre'].cpu() for layer in range(model.cfg.n_layers)]),
        'layer batch pos dim -> batch layer pos dim'
    )
    norm_acts = acts / acts.norm(dim=-1, keepdim=True)
    return norm_acts

def get_coref_scores(model, ds, chat_style, form_path, form_type, affinity_fn, affinity_fn_kwargs):
    affinity_fn = lkp.get_affinity_fn(
        affinity_fn=affinity_fn,
        form_path=form_path,
        form_type=form_type,
        affinity_fn_kwargs=affinity_fn_kwargs
    )
    ds = cge.templatize_winobias(ds, strict=False)
    wb_template = tb.WinobiasTemplate('llama')
    all_pronoun_scores = []
    all_distractor_scores = []
    for batch_slice in SliceRange(0, len(ds), 64):
        batch = ds[batch_slice]
        input_tokens = []
        context_span = []
        all_token_maps = []
        for row in batch:
            tokens, token_maps, answers = ta.generate_tokenized_prompts(
                template=wb_template,
                template_content=dict(query='', prefix='', prompt_style=chat_style),
                context_content=[{
                    k: row[k]
                    for k in ['template', 'subject', 'distractor', 'pronoun']
                }],
                prompt_id=0,
                num_answers=0
            )
            input_tokens.append(tokens)
            context_span.append(token_maps['context_section'])
            all_token_maps.append(token_maps)
        input_tokens, pad_lengths = pad_right(input_tokens)

        _, source_cache = act_loc.get_normal_cache(
            model,
            input_tokens=input_tokens,
            prompt_ids=None, # not used
            stacked_token_maps=None,
            answer_tokens=None,
        )
        norm_acts = cache_to_norm_acts(model, source_cache)
        mask = torch.zeros_like(input_tokens, dtype=torch.bool)
        for i, (s, e) in enumerate(context_span):
            mask[i, s:e] = True

        scores = affinity_fn(norm_acts.float())

        pronoun_pos = torch.tensor([tm['context'][0]['pronoun'].end-1 for tm in all_token_maps])
        distractor_pos = torch.tensor([tm['context'][0]['distractor'].end-1 for tm in all_token_maps])
        subject_pos = torch.tensor([tm['context'][0]['subject'].end-1 for tm in all_token_maps])

        all_pronoun_scores.append(scores[torch.arange(len(batch)), subject_pos, pronoun_pos])

        all_distractor_scores.append(scores[torch.arange(len(batch)), distractor_pos, pronoun_pos])
    all_pronoun_scores = torch.cat(all_pronoun_scores)
    all_distractor_scores = torch.cat(all_distractor_scores)
    return all_pronoun_scores, all_distractor_scores
        