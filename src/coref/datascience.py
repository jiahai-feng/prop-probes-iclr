import os
import copy
from collections import OrderedDict
from coref import COREF_ROOT
from functools import partial

import numpy as np
import torch
import einops
from einops import rearrange, einsum

from transformer_lens.utilities.devices import get_device_for_block_index
import transformer_lens.HookedTransformer as HookedTransformer
import transformer_lens.utils as utils

from coref.interventions import run_cached_model


from coref.datasets.templates.common import recursify, rotate


from coref.parameters import (
    build_example,
)


def collect_data(
    *,
    model,
    num_samples=500,
    batch_size=50,
    template,
    context_content,
    template_content,
    num_answers,
    prompt_id_start,
    execution_fn=None,
    extractor_map,
    hook_name="resid_pre",
    canonize_token_maps=True,
):
    """
    Args:
        extractor_map : token_maps -> Dict[... List[pos: int]]
    """
    is_hf = not isinstance(model, HookedTransformer)
    if execution_fn is None:
        if is_hf:
            model_runner = (
                lambda input_tokens, prompt_ids, stacked_token_maps: run_cached_model(
                    execution_fn=lambda: model(input_tokens).logits,
                    model=model,
                    names_filter=lambda x: hook_name in x,
                )
            )
        else:
            model_runner = lambda input_tokens, prompt_ids, stacked_token_maps: model.run_with_cache(
                input_tokens, names_filter=lambda x: "hook_" + hook_name in x
            )
    else:
        model_runner = (
            lambda input_tokens, prompt_ids, stacked_token_maps: run_cached_model(
                execution_fn=lambda: execution_fn(
                    input_tokens, prompt_ids, stacked_token_maps
                ),
                model=model,
                names_filter=lambda x: "hook_" + hook_name in x,
            )
        )
    all_acts = []
    all_tokens = []
    all_answers = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        input_tokens, answer_tokens, stacked_token_maps = build_example(
            batch_size=end - start,
            template=template,
            template_content=template_content,
            context_content=context_content,
            prompt_id_start=prompt_id_start + start,
            num_answers=num_answers,
        )
        prompt_ids = slice(prompt_id_start + start, prompt_id_start + end)
        logits, cache = model_runner(input_tokens, prompt_ids, stacked_token_maps)

        if canonize_token_maps:
            parsed_token_maps = template.canonize_token_maps(stacked_token_maps)
        else:
            parsed_token_maps = stacked_token_maps
        actual_selector = extractor_map(parsed_token_maps)
        if hook_name == "resid_pre":
            acts = recursify(
                lambda positions: torch.stack(
                    [
                        (
                            lambda activations: activations.cpu().gather(
                                dim=1,
                                index=einops.repeat(
                                    positions,
                                    "batch -> batch dummy dim",
                                    dummy=1,
                                    dim=activations.shape[2],
                                ),
                            )[:, 0, :]
                        )(cache[utils.get_act_name(hook_name, layer)])
                        for layer in range(model.cfg.n_layers)
                    ]
                ),
                dtype=torch.Tensor,
            )(actual_selector)
        elif hook_name == "z":
            acts = recursify(
                lambda positions: torch.stack(
                    [
                        (
                            lambda z: z.cpu().gather(  # [batch, pos, head_index, d_head]
                                dim=1,
                                index=einops.repeat(
                                    positions,
                                    "batch -> batch dummy head_index dim",
                                    dummy=1,
                                    head_index=z.shape[2],
                                    dim=z.shape[3],
                                ),
                            )[
                                :, 0, :, :
                            ]
                        )(cache[utils.get_act_name(hook_name, layer)])
                        for layer in range(model.cfg.n_layers)
                    ]
                ),
                dtype=torch.Tensor,
            )(actual_selector)
        elif hook_name == "pattern" or hook_name == "attn_scores":
            acts = recursify(
                lambda positions: torch.stack(
                    [
                        (
                            lambda pattern: pattern.cpu().gather(  # [batch, head_index, query_pos, key_pos]
                                dim=2,
                                index=einops.repeat(
                                    positions,
                                    "batch -> batch head_index dummy dim",
                                    dummy=1,
                                    head_index=pattern.shape[1],
                                    dim=pattern.shape[3],
                                ),
                            )[
                                :, :, 0, :
                            ]
                        )(cache[utils.get_act_name(hook_name, layer)])
                        for layer in range(model.cfg.n_layers)
                    ]
                ),
                dtype=torch.Tensor,
            )(actual_selector)

        else:
            assert False, f"hook_name {hook_name} not implemented"
        tokens = recursify(
            lambda positions: input_tokens.cpu().gather(
                dim=1, index=positions[:, None]
            )[:, 0],
            dtype=torch.Tensor,
        )(actual_selector)
        all_acts.append(acts)
        all_tokens.append(tokens)
        all_answers.append(answer_tokens)
    if hook_name == "resid_pre":
        all_acts = rotate(
            lambda x: rearrange(
                torch.stack(x), "batch layer minibatch dim->(batch minibatch) layer dim"
            ),
            dtype=torch.Tensor,
        )(all_acts)
    elif hook_name == "pattern" or hook_name == "attn_scores":
        all_acts = rotate(
            lambda x: rearrange(
                torch.stack(x),
                "batch layer minibatch headidx dim->(batch minibatch) layer headidx dim",
            ),
            dtype=torch.Tensor,
        )(all_acts)
    elif hook_name == "z":
        all_acts = rotate(
            lambda x: rearrange(
                torch.stack(x),
                "batch layer minibatch headidx dim->(batch minibatch) layer headidx dim",
            ),
            dtype=torch.Tensor,
        )(all_acts)

    all_tokens = rotate(lambda x: torch.concatenate(x), dtype=torch.Tensor)(all_tokens)
    all_answers = rotate(lambda x: torch.concatenate(x), dtype=torch.Tensor)(
        all_answers
    )
    return all_acts, all_tokens, all_answers
