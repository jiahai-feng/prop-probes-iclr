import torch
import numpy as np
from fancy_einsum import einsum
import einops

from jaxtyping import Float, Int, Bool
from typing import List, Union, Optional
from functools import partial
from transformer_lens import HookedTransformer
from coref.hf_interventions import cache_resid_pre, hook_resid_pre, hook_rot_k, hook_model

import transformer_lens.utils as utils


def compose_hook_fns(hook_fns):
    def wrapper(tensor, hook, **kwargs):
        for hook_fn in hook_fns:
            tensor = hook_fn(tensor, hook, **kwargs)
        return tensor

    return wrapper


def patch_masked_residue(
    target_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    mask: Bool[torch.Tensor, "batch pos"],
    source_cache,
):
    device = target_residual_component.device
    target_residual_component[mask.to(device), :] = (
        source_cache[hook.name].to(device)[mask.to(device), :].to(device)
    )
    return target_residual_component


def rotary_deltas(
    x: Float[torch.Tensor, "batch pos head_index d_head"],
    pos_deltas: Int[torch.Tensor, "batch pos"],
    attn,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    # adapted from components.py -> Attention -> apply_rotary
    x_pos = x.size(1)
    x_rot = x[..., : attn.cfg.rotary_dim]  # batch pos head_index d_head
    x_pass = x[..., attn.cfg.rotary_dim :]
    x_flip = attn.rotate_every_two(x_rot)
    abs_pos = torch.abs(pos_deltas)
    coses = attn.rotary_cos[abs_pos]  # batch pos d_head
    sines = attn.rotary_sin[abs_pos] * torch.sign(pos_deltas)[..., None]
    x_rotated = x_rot * einops.rearrange(
        coses, "b p d -> b p 1 d"
    ) + x_flip * einops.rearrange(sines, "b p d -> b p 1 d")
    return torch.cat([x_rotated, x_pass], dim=-1)


def patch_rotary_k(
    target_k: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    pos_deltas: Int[torch.Tensor, "batch pos"],
    rotate_function,
):
    # consistency tests:
    # y = rotate_function(target_k, pos_deltas)
    # x = rotate_function(y, -pos_deltas)
    # assert torch.allclose(target_k, x, rtol=1e-3, atol=1e-4)
    return rotate_function(target_k, pos_deltas.to(target_k.device))


def run_intervened_model(
    model,
    source_tokens,
    layer_hook_fns,
    dependent_start,
    pos_deltas_fn=None,
):
    """
    Args:
        layer_hook_fns: Dict[
            layer: int,
            hook_fn : (tensor: Tensor[batch, pos, d_model], hook: nn.Module)
                            -> (tensor: Tensor[batch, pos, d_model])
        ]
        pos_deltas_fn: (pos_deltas) -> pos_deltas (Int[torch.Tensor, "batch layer pos"])
            Interpret pos_deltas as the position you want to virtually add
            Okay to modify in-place
        dependent_start : [batch] | int
    Returns:
        intervened_logits : [batch, pos, vocab_size]
    """
    is_hf = not isinstance(model, HookedTransformer)
    pos_deltas = torch.zeros(
        (source_tokens.shape[0], model.cfg.n_layers, source_tokens.shape[1]),
        dtype=torch.int64,
    )
    if isinstance(dependent_start, int):
        dependent_start = torch.tensor([dependent_start] * source_tokens.shape[0])
    assert dependent_start.shape[0] == source_tokens.shape[0]
    assert len(dependent_start.shape) == 1

    if pos_deltas_fn is not None:
        pos_deltas = pos_deltas_fn(pos_deltas)

    target_dependent_mask = torch.zeros(
        (source_tokens.shape[0], model.cfg.n_layers, source_tokens.shape[1]), dtype=bool
    )
    for batch in range(source_tokens.shape[0]):
        target_dependent_mask[batch, :, dependent_start[batch] :] = True
    source_mask = ~target_dependent_mask  # batch, layer, pos

    with torch.no_grad():  # shouldn't make a diff, because we already detach our cache
        if is_hf:
            with cache_resid_pre(model) as source_cache:
                source_logits = model(source_tokens).logits
        else:
            source_logits, source_cache = model.run_with_cache(
                source_tokens, names_filter=lambda x: any(y in x for y in ["resid_pre"])
            )
    resid_pre_hooks = []
    rot_hooks = []
    for layer in range(model.cfg.n_layers):
        hook_fns = [
            partial(
                patch_masked_residue,
                mask=source_mask[:, layer, :],
                source_cache=source_cache,
            )
        ]

        if layer in layer_hook_fns:
            hook_fns.append(layer_hook_fns[layer])

        resid_pre_hooks.append(
            (utils.get_act_name("resid_pre", layer), compose_hook_fns(hook_fns))
        )
        if not is_hf:
            if pos_deltas_fn is not None:
                rot_hooks.append(
                    (
                        utils.get_act_name("rot_k", layer),
                        partial(
                            patch_rotary_k,
                            pos_deltas=pos_deltas[:, layer, :],
                            rotate_function=partial(
                                rotary_deltas, attn=model.blocks[layer].attn
                            ),
                        ),
                    )
                )
    if is_hf:
        with hook_resid_pre(model, resid_pre_hooks, prepend=True):
            if pos_deltas_fn is not None:
                with hook_rot_k(model, pos_deltas):
                    intervened_logits = model(source_tokens).logits
            else:
                intervened_logits = model(source_tokens).logits

    else:
        with hook_model(model, resid_pre_hooks + rot_hooks, prepend=True):
            intervened_logits = model(source_tokens)
    return intervened_logits


def run_cached_model(
    execution_fn,
    model,
    names_filter=None,
    device=None,
    remove_batch_dim=False,
    incl_bwd=False,
    reset_hooks_end=True,
    clear_contexts=False,
    extra_bwd_hooks=[],
):
    """
    Modified from HookedRootModule.run_with_cache
    Args:
        execution_fn : () -> logits
    Return:
        (logits, cache_dict)
    """
    is_hf = not isinstance(model, HookedTransformer)

    if is_hf:
        with cache_resid_pre(model) as cache:
            model_out = execution_fn()
        assert not incl_bwd
        return model_out, cache
    else:
        cache_dict, fwd, bwd = model.get_caching_hooks(
            names_filter, incl_bwd, device, remove_batch_dim=remove_batch_dim
        )

        with model.hooks(
            fwd_hooks=fwd,
            bwd_hooks=bwd + extra_bwd_hooks,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            model_out = execution_fn()
            if incl_bwd:
                # model_out.backward()
                pass  # disable this. we expect execution_fn() to do the backward

        return model_out, cache_dict
