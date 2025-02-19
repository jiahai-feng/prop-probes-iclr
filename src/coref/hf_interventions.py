"""
Hooks implemented directly on HF llama models.

This should be compatible with mistral, but not with pythia (which uses gpt neo).

Relevant HF code:
- [Llama Implementation](https://github.com/huggingface/transformers/blob/bd50402b56980ff17e957342ef69bd9b0dd45a7b/src/transformers/models/llama/modeling_llama.py)
- [Mistral Implementation](https://github.com/huggingface/transformers/blob/bd50402b56980ff17e957342ef69bd9b0dd45a7b/src/transformers/models/mistral/modeling_mistral.py)
"""

from contextlib import contextmanager
from functools import partial

import torch
import einops

resid_pre_namer = lambda layer: f"blocks.{layer}.hook_resid_pre"


def get_llama_resid_pre(model):
    return model.model.layers


def get_tlens_resid_pre(model):
    return [block.hook_resid_pre for block in model.blocks]


def get_resid_pre(model):
    if hasattr(model, "blocks"):
        return get_tlens_resid_pre(model)
    else:
        return get_llama_resid_pre(model)


@contextmanager
def cache_resid_pre(model, cache=None, detach=True):
    if cache is None:
        cache = {}

    def cache_hook(module, args, *, layer):
        # see https://github.com/huggingface/transformers/blob/bd50402b56980ff17e957342ef69bd9b0dd45a7b/src/transformers/models/llama/modeling_llama.py#L653
        (hidden_states,) = args  # batch, seq_len, embed_dim
        if detach:
            cache[
                resid_pre_namer(layer)
            ] = hidden_states.clone().detach()  # we follow transformer lens convention
        else:
            cache[resid_pre_namer(layer)] = hidden_states.clone()
        return args

    hook_handles = []
    for i, block in enumerate(get_resid_pre(model)):
        handle = block.register_forward_pre_hook(partial(cache_hook, layer=i))
        hook_handles.append(handle)
    yield cache
    for handle in hook_handles:
        handle.remove()


'''
write a hook context manager that deals with arbitrary names compatible with HookedTransformer

If HF model, map subset of these names to correct HF names
'''

import re
def hooked_to_hf(hooked_name):
    # match resid_pre
    match = re.fullmatch(r"blocks\.(\d+)\.hook_resid_pre", hooked_name)
    if match is not None:
        return f"model.layers.{match[1]}"
    return None

def lookup_module(model, hook_name):
    if hasattr(model, "blocks"):
        # assume model is a HookedTransformer
        d = dict(model.named_modules())
        return d[hook_name] if hook_name in d else None
    else:
        # assume HF model
        # map resid_pre:
        hf_name = hooked_to_hf(hook_name)
        if hf_name is None:
            return None
        d = dict(model.named_modules())
        return d[hf_name] if hf_name in d else None
        

from functools import wraps
@contextmanager
def hook_model(model, hooks, prepend=False, hf_name=False):
    hook_handles = []

    def flattened_hook_fn(module, args, *, hook_fn, hook_name):
        (hidden_states,) = args

        class Hook:
            name = hook_name
            mod = module

        return hook_fn(hidden_states, Hook)

    if hf_name:
        name_map = dict(model.named_modules())
    for hook_name, hook_fn in hooks:
        if hf_name:
            mod = name_map[hook_name]
        else:
            mod = lookup_module(model, hook_name)
        if mod is None:
            raise Exception(f"No submodule {hook_name} found!")
    try:
        if prepend:
            hooks = hooks[::-1]
        for hook_name, hook_fn in hooks:
            if hf_name:
                mod = name_map[hook_name]
            else:
                mod = lookup_module(model, hook_name)
            handle = mod.register_forward_pre_hook(
                wraps(hook_fn)(partial(flattened_hook_fn, hook_fn=hook_fn, hook_name=hook_name)),
                prepend=prepend
            )
            hook_handles.append(handle)
        yield model
    finally:
        for handle in hook_handles:
            handle.remove()

@contextmanager
def hook_model_bwd(model, hooks, prepend=False):
    hook_handles = []

    def flattened_hook_fn(module, module_output, *, hook_fn, hook_name):
        (hidden_states,) = module_output

        class Hook:
            name = hook_name
            mod = module

        return hook_fn(hidden_states, Hook)

    for hook_name, hook_fn in hooks:
        mod = lookup_module(model, hook_name)
        if mod is None:
            raise Exception(f"No submodule {hook_name} found!")
    try:
        if prepend:
            hooks = hooks[::-1]
        for hook_name, hook_fn in hooks:
            mod = lookup_module(model, hook_name)
            handle = mod.register_full_backward_pre_hook(
                wraps(hook_fn)(partial(flattened_hook_fn, hook_fn=hook_fn, hook_name=hook_name)),
                prepend=prepend
            )
            hook_handles.append(handle)
        yield model
    finally:
        for handle in hook_handles:
            handle.remove()

hook_resid_pre = hook_model


@contextmanager
def hook_rot_k(model, pos_deltas):
    '''
    This messes with the position_ids input to HF's forward function.
    There is no analog for HookedTransformers
    '''
    hook_handles = []

    def rotate_fn(module, args, kwargs, *, pos_deltas):
        (hidden_states,) = args
        if "position_ids" in kwargs and kwargs["position_ids"] is not None:
            old_position_ids = kwargs["position_ids"]
        else:
            batch, seq, dim = hidden_states.shape
            old_position_ids = einops.repeat(torch.arange(0, seq), "s -> b s", b=batch)
        position_ids = old_position_ids + pos_deltas.to(old_position_ids.device)
        kwargs["position_ids"] = position_ids
        return (args, kwargs)

    try:
        for i, block in enumerate(model.model.layers):
            handle = block.register_forward_pre_hook(
                partial(rotate_fn, pos_deltas=pos_deltas[:, i, :]), with_kwargs=True,
            )
            hook_handles.append(handle)
        yield model
    finally:
        for handle in hook_handles:
            handle.remove()


def clear_resid_pre(model):
    """
    For use when hooks are not reset cleanly
    """
    for layer in model.model.layers:
        while layer._forward_pre_hooks:
            layer._forward_pre_hooks.popitem()
