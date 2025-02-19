import os
from coref import COREF_ROOT
from coref.utils import cache
from functools import partial
import itertools
import gc

import torch
import pandas as pd

import coref.models as models
import coref.experiments.triplet_expts as te
import coref.datasets.templates.triplet as tt
from functools import partial
import numpy as np
import torch
import einops
import torch.nn.functional as F

import coref.datasets.templates.simple as ts
import coref.datasets.templates.common as tc
import coref.datasets.templates.triplet as tt
import coref.datasets.api as ta
import coref.parameters as p
import coref.datascience as ds
import coref.expt_utils as eu

import coref.interventions as cint
import coref.injection_interventions as ii

import coref.flow as flow
import coref.eval_subspace as ess
from tqdm.notebook import trange, tqdm

_model = None
_train_template = None
_prompt_type = None
_switch_interventions, _mean_params = (None, None)


def init(
    model,
    train_template,
    prompt_type,
):
    global _model, _train_template, _prompt_type, _switch_interventions, _mean_params
    _model = model
    _train_template = train_template
    _prompt_type = prompt_type
    _switch_interventions, _mean_params = build_interventions()


def dim_model_grads(model):
    for param in model.parameters():
        param.requires_grad = False
    model.embed.W_E.requires_grad = True


def build_interventions():
    num_mean_samples = 100
    train_prompt_id_start = 0
    category_widths = {"name": 2, "attr": 1}
    with torch.no_grad():
        both_acts = [
            ds.collect_data(
                model=_model,
                num_samples=num_mean_samples,
                batch_size=20,
                template=_train_template,
                template_content=dict(query_name=0, prompt_type=_prompt_type),
                context_content=ctx,
                num_answers=2,
                prompt_id_start=train_prompt_id_start,
                extractor_map=lambda token_maps: {
                    "name": [
                        [x + i for i in range(category_widths["name"])]
                        for x in token_maps["name"]
                    ],
                    "attr": [
                        [x + i for i in range(category_widths["attr"])]
                        for x in token_maps["attr"]
                    ],
                },
            )[
                0
            ]  # returns acts, tokens, ans by default
            for ctx in [
                [
                    tc.Statement(0, 0, None),
                    tc.Statement(1, 1, None),
                ],
                [
                    tc.Statement(1, 1, None),
                    tc.Statement(0, 0, None),
                ],
            ]
        ]

        means = {
            attr: p.build_mean_param(
                _model,
                both_acts[1][attr][1],
                both_acts[0][attr][0],
            )
            for attr in both_acts[0]
        }  #  batch layer dim
        # this gives b(1) - b(0)
        injections = {
            category: p.compose_interventions(
                *[
                    ii.injection_intervention_from_param(
                        model=_model,
                        param=means[category],
                        width=category_widths[category],
                        category=category,
                        target_side=side,
                        adapter=ii.scale_adapter(
                            means[category].default_adapter,
                            -1 if side else 1,
                        ),
                    )
                    for side in [0, 1]
                ]
            )
            for category in ["name", "attr"]
        }
    return injections, means


def normal_run(input_tokens, prompt_ids, stacked_token_maps, answer_tokens):
    return _model(input_tokens)


def swap_category_run(
    input_tokens, prompt_ids, stacked_token_maps, answer_tokens, category
):
    return cint.run_intervened_model(
        model=_model,
        source_tokens=input_tokens,
        layer_hook_fns=p.interventions_to_hook_fns(
            interventions=_switch_interventions[category],
            prompt_ids=prompt_ids,
            parsed_token_maps=_train_template.canonize_token_maps(stacked_token_maps),
        ),
        dependent_start=stacked_token_maps["context_section"][:, 1],
        pos_deltas_fn=None,
    )


def swap_category_run_exogenous(
    input_tokens, prompt_ids, stacked_token_maps, answer_tokens, category
):
    return cint.run_intervened_model(
        model=_model,
        source_tokens=input_tokens,
        layer_hook_fns=p.interventions_to_hook_fns(
            interventions=_switch_interventions[category],
            prompt_ids=prompt_ids,
            parsed_token_maps=_train_template.canonize_token_maps(stacked_token_maps),
        ),
        dependent_start=input_tokens.shape[1],  # key difference: no dependent_start
        pos_deltas_fn=None,
    )


def get_exogenous_cache(
    input_tokens, prompt_ids, stacked_token_maps, answer_tokens, category
):
    logits, exo_cache = cint.run_cached_model(  # [batch, pos, head_index, d_head]
        execution_fn=lambda: swap_category_run_exogenous(
            input_tokens=input_tokens,
            prompt_ids=prompt_ids,
            stacked_token_maps=stacked_token_maps,
            answer_tokens=answer_tokens,
            category=category,
        ),
        model=_model,
        names_filter=lambda x: "hook_z" in x,
    )
    return logits.detach(), exo_cache


def get_normal_cache(input_tokens, prompt_ids, stacked_token_maps, answer_tokens):
    logits, normal_cache = cint.run_cached_model(  # [batch, pos, head_index, d_head]
        execution_fn=lambda: normal_run(
            input_tokens=input_tokens,
            prompt_ids=prompt_ids,
            stacked_token_maps=stacked_token_maps,
            answer_tokens=answer_tokens,
        ),
        model=_model,
        names_filter=lambda x: "hook_z" in x,
        incl_bwd=False,
    )
    return logits.detach(), normal_cache


def get_abnormal_cache(
    input_tokens, prompt_ids, stacked_token_maps, answer_tokens, hook_context, category
):
    cache = {}

    def cache_forward(tensor, hook):
        cache[hook.name] = tensor

    def cache_backward(tensor, hook):
        cache[hook.name + "_grad"] = tensor

    def block_qk_hook(grad, hook):
        if hook_context["block_qk"]:
            ret = torch.zeros_like(grad)
            for batch in range(ret.shape[0]):
                ret[
                    batch, :, :, : stacked_token_maps["context_section"][batch, 1]
                ] = grad[batch, :, :, : stacked_token_maps["context_section"][batch, 1]]
            return (ret,)

    def block_attn_hook(grad, hook):
        if hook_context["block_attn"]:
            ret = torch.zeros_like(grad)
            return (ret,)

    with torch.enable_grad():
        with _model.hooks(
            fwd_hooks=[
                (f"blocks.{layer}.attn.hook_z", cache_forward)
                for layer in range(_model.cfg.n_layers)
            ],
            bwd_hooks=(
                [
                    (f"blocks.{layer}.attn.hook_z", cache_backward)
                    for layer in range(_model.cfg.n_layers)
                ]
                + [
                    (f"blocks.{layer}.attn.hook_pattern", block_qk_hook)
                    for layer in range(_model.cfg.n_layers)
                ]
                + [
                    (f"blocks.{layer}.attn.hook_z", block_attn_hook)
                    for layer in range(_model.cfg.n_layers)
                ]
            ),
        ):
            logits = swap_category_run(
                input_tokens=input_tokens,
                prompt_ids=prompt_ids,
                stacked_token_maps=stacked_token_maps,
                answer_tokens=answer_tokens,
                category=category,
            )
    return logits, cache


def stack_cache(cache_list):
    return {
        key: torch.concatenate([cache[key].cpu() for cache in cache_list])
        for key in cache_list[0]
    }


def extract_top_k_head_pos(saliency_map):
    saliency_map = saliency_map.cpu().numpy()
    all_heads = []
    for pos in range(saliency_map.shape[0]):
        for layer in range(saliency_map.shape[1]):
            for head in range(saliency_map.shape[2]):
                all_heads.append(((layer, head, pos), saliency_map[pos, layer, head]))
    all_heads = sorted(all_heads, key=lambda x: -x[1])
    return all_heads


def wipe_backwards(cache):
    for key in list(cache.keys()):
        if "_grad" in key:
            del cache[key]


def discover_terminal(
    input_tokens,
    prompt_ids,
    stacked_token_maps,
    answer_tokens,
    answer,
    category,
    terminal_threshold,
    block_qk=True,
    loss="logit_diff",
    suppress_context=True,
):
    """
    returns masked_total_score, direct_score, total_score
    each of which is a tensor of shape: pos layer head_idx

    The masked_total_score the same as total_score, but masked based on whether direct_score is a 
    significant portion of total_score (thresholded using terminal_threshold)

    There is a edge case where abnormal and exogenous cache can have non-zero forward diff in the context,
    because at the token positions where we intervene, the hook_z gets the updated version, even though this
    never gets propagated. We just artificically set the scores to zero as a hacky patch.
    """
    dim_model_grads(_model)
    hook_context = {}
    _, exogenous_cache = get_exogenous_cache(
        input_tokens=input_tokens,
        prompt_ids=prompt_ids,
        stacked_token_maps=stacked_token_maps,
        answer_tokens=answer_tokens,
        category=category,
    )
    _, normal_cache = get_normal_cache(
        input_tokens=input_tokens,
        prompt_ids=prompt_ids,
        stacked_token_maps=stacked_token_maps,
        answer_tokens=answer_tokens,
    )
    logits, abnormal_cache = get_abnormal_cache(
        input_tokens=input_tokens,
        prompt_ids=prompt_ids,
        stacked_token_maps=stacked_token_maps,
        answer_tokens=answer_tokens,
        hook_context=hook_context,
        category=category,
    )

    forward_name = lambda layer: f"blocks.{layer}.attn.hook_z"
    backward_name = lambda layer: f"blocks.{layer}.attn.hook_z_grad"

    # relax root without blocking attn

    hook_context["block_attn"] = False
    hook_context["block_qk"] = block_qk
    wipe_backwards(abnormal_cache)

    with torch.enable_grad():
        if loss == "logit_diff":
            logit_diff = (
                logits[torch.arange(logits.shape[0]), -1, answer_tokens[:, answer]]
                - logits[
                    torch.arange(logits.shape[0]), -1, answer_tokens[:, 1 - answer]
                ]
            )
            assert logit_diff.shape == (logits.shape[0],)
            logit_diff.mean().backward(retain_graph=True)
        elif loss == "log_prob":
            log_prob = logits[
                torch.arange(logits.shape[0]), -1, answer_tokens[:, answer]
            ] - torch.logsumexp(logits[:, -1, :], dim=1, keepdim=False)
            assert log_prob.shape == (logits.shape[0],)
            log_prob.mean().backward(retain_graph=True)
        else:
            assert False, f"Unknown loss type {loss}"

    total_score = einops.rearrange(
        torch.stack(
            [
                einops.einsum(
                    abnormal_cache[forward_name(layer)].detach()
                    - normal_cache[forward_name(layer)],
                    abnormal_cache[backward_name(layer)].detach()
                    if backward_name(layer) in abnormal_cache
                    else torch.zeros_like(abnormal_cache[forward_name(layer)]),
                    "batch pos head_idx d_head, batch pos head_idx d_head -> batch pos head_idx",
                ).cpu()
                for layer in range(_model.cfg.n_layers)
            ]
        ),
        "layer batch pos head_idx -> batch pos layer head_idx",
    ).sum(dim=0)
    direct_score = einops.rearrange(
        torch.stack(
            [
                einops.einsum(
                    exogenous_cache[forward_name(layer)].detach()
                    - normal_cache[forward_name(layer)],
                    abnormal_cache[backward_name(layer)].detach()
                    if backward_name(layer) in abnormal_cache
                    else torch.zeros_like(abnormal_cache[forward_name(layer)]),
                    "batch pos head_idx d_head, batch pos head_idx d_head -> batch pos head_idx",
                ).cpu()
                for layer in range(_model.cfg.n_layers)
            ]
        ),
        "layer batch pos head_idx -> batch pos layer head_idx",
    ).sum(dim=0)

    # hacky patch
    if suppress_context:
        print("suppressing context")
        total_score[: stacked_token_maps["context_section"][0, 1]] = 0
        direct_score[: stacked_token_maps["context_section"][0, 1]] = 0

    is_terminal = (direct_score > terminal_threshold * total_score) & (total_score > 0)
    masked_total_score = torch.where(is_terminal, total_score, 0)

    # Kill backward graph
    with torch.enable_grad():
        logits.mean().backward()

    return masked_total_score, direct_score, total_score


def discover_circuit(
    input_tokens,
    prompt_ids,
    stacked_token_maps,
    answer_tokens,
    answer,
    category,
    loss="logit_diff",
    relax_mode="dag",
):
    """
    category: 'name' | 'attr'
    loss: 'logit_diff' | 'log_prob'
    relax_mode: 'dag' | 'staged'

    If relax_mode is 'dag', returns:
        Dict[ head: (layer, head_idx, pos) -> {
            'score': Tensor(float)
            'children': List [( child_head: (layer, head_idx, pos), child_score ) ]
        }]
    If relax_mode is 'staged', returns:
        attn_head_at_stage, terminal_heads_at_stage
        each of this is a list num_stages long
        - each element of this list is the tuple:
         ( (head, layer, pos), Dict['score': torch.tensor[], 'vec': torch.tensor[head_dim])])

    """
    hook_context = {}
    _, exogenous_cache = get_exogenous_cache(
        input_tokens=input_tokens,
        prompt_ids=prompt_ids,
        stacked_token_maps=stacked_token_maps,
        answer_tokens=answer_tokens,
        category=category,
    )
    _, normal_cache = get_normal_cache(
        input_tokens=input_tokens,
        prompt_ids=prompt_ids,
        stacked_token_maps=stacked_token_maps,
        answer_tokens=answer_tokens,
    )
    logits, abnormal_cache = get_abnormal_cache(
        input_tokens=input_tokens,
        prompt_ids=prompt_ids,
        stacked_token_maps=stacked_token_maps,
        answer_tokens=answer_tokens,
        hook_context=hook_context,
        category=category,
    )

    forward_name = lambda layer: f"blocks.{layer}.attn.hook_z"
    backward_name = lambda layer: f"blocks.{layer}.attn.hook_z_grad"

    def relax_root():
        hook_context["block_attn"] = True
        hook_context["block_qk"] = True
        wipe_backwards(abnormal_cache)
        with torch.enable_grad():
            if loss == "logit_diff":
                logit_diff = (
                    logits[torch.arange(logits.shape[0]), -1, answer_tokens[:, answer]]
                    - logits[
                        torch.arange(logits.shape[0]), -1, answer_tokens[:, 1 - answer]
                    ]
                )
                assert logit_diff.shape == (logits.shape[0],)
                logit_diff.mean().backward(retain_graph=True)
            elif loss == "log_prob":
                log_prob = logits[
                    torch.arange(logits.shape[0]), -1, answer_tokens[:, answer]
                ] - torch.logsumexp(logits[:, -1, :], dim=1, keepdim=False)
                assert log_prob.shape == (logits.shape[0],)
                log_prob.mean().backward(retain_graph=True)
            else:
                assert False, f"Unknown loss type {loss}"

    def relax(layer, head_idx, query_pos, vec):
        hook_context["block_attn"] = True
        hook_context["block_qk"] = True
        wipe_backwards(abnormal_cache)
        with torch.enable_grad():
            loss = einops.einsum(
                abnormal_cache[forward_name(layer)][:, query_pos, head_idx, :],
                vec.to(abnormal_cache[forward_name(layer)].device),
                "batch dim, batch dim -> batch",
            )
            loss.sum().backward(retain_graph=True)

    def compute_saliency():
        saliency = einops.rearrange(
            torch.stack(
                [
                    einops.einsum(
                        abnormal_cache[forward_name(layer)].detach()
                        - normal_cache[forward_name(layer)],
                        abnormal_cache[backward_name(layer)].detach()
                        if backward_name(layer) in abnormal_cache
                        else torch.zeros_like(abnormal_cache[forward_name(layer)]),
                        "batch pos head_idx d_head, batch pos head_idx d_head -> batch pos head_idx",
                    ).cpu()
                    for layer in range(_model.cfg.n_layers)
                ]
            ),
            "layer batch pos head_idx -> batch pos layer head_idx",
        )
        return saliency.sum(dim=0)  # pos layer head_idx

    def relax_node(attn_head, vec, is_root=False):
        if is_root:
            relax_root()
        else:
            relax(*attn_head, vec)
        saliency = compute_saliency()
        top_k_heads = extract_top_k_head_pos(saliency)[:branching_factor]
        return [
            dict(
                attn_head=(layer, head, pos),
                score=score,
                vec=abnormal_cache[backward_name(layer)][:, pos, head, :].detach(),
            )
            for (layer, head, pos), score in top_k_heads
            if score > 0
        ]

    def check_terminal(attn_head, vec):
        layer, head_idx, pos = attn_head
        total_score = einops.einsum(
            abnormal_cache[forward_name(layer)].detach()[:, pos, head_idx, :]
            - normal_cache[forward_name(layer)][:, pos, head_idx, :],
            vec.to(normal_cache[forward_name(layer)].device),
            "batch d_head, batch d_head -> batch",
        ).sum(dim=0)
        direct_score = einops.einsum(
            exogenous_cache[forward_name(layer)].detach()[:, pos, head_idx, :]
            - normal_cache[forward_name(layer)][:, pos, head_idx, :],
            vec.to(normal_cache[forward_name(layer)].device),
            "batch d_head, batch d_head -> batch",
        ).sum(dim=0)
        assert total_score > 0
        return (direct_score > (terminal_threshold * total_score), direct_score)

    def cleanup():
        # Kill backward graph
        with torch.enable_grad():
            logits.mean().backward(retain_graph=False)

    if relax_mode == "staged":
        terminal_threshold = 0.5
        branching_factor = 50
        max_nodes_per_stage = 50
        num_stages = 5

        def stage_relax(attn_head_list, is_root=False):
            if is_root:
                all_next_heads = [relax_node(attn_head=None, vec=None, is_root=True)]
            else:
                all_next_heads = [
                    relax_node(attn_head=attn_head, vec=vec, is_root=False)
                    for attn_head, vec in attn_head_list
                ]
            aggregate_heads = {}

            def proc(attn_head, score, vec):
                if attn_head in aggregate_heads:
                    aggregate_heads[attn_head]["score"] += score
                    aggregate_heads[attn_head]["vec"] += vec
                else:
                    aggregate_heads[attn_head] = dict(score=score, vec=vec)

            for next_heads in all_next_heads:
                for head in next_heads:
                    proc(**head)
            top_heads = sorted(
                list(aggregate_heads.items()), key=lambda x: -x[1]["score"]
            )
            return top_heads[:max_nodes_per_stage]

        attn_head_at_stage = []
        terminal_heads_at_stage = []
        for stage in trange(num_stages):
            if stage == 0:
                attn_head_list = stage_relax(attn_head_list=None, is_root=True)
            else:
                attn_head_list = stage_relax(
                    attn_head_list=[
                        (attn_head, d["vec"])
                        for (attn_head, d), is_term in zip(
                            attn_head_at_stage[-1], is_terminal
                        )
                        if not is_term
                    ],
                    is_root=False,
                )
            attn_head_at_stage.append(attn_head_list)
            is_terminal = [
                check_terminal(attn_head, d["vec"])[0]
                for attn_head, d in attn_head_at_stage[-1]
            ]
            terminal_heads_at_stage.append(
                [
                    tup
                    for tup, is_term in zip(attn_head_at_stage[-1], is_terminal)
                    if is_term
                ]
            )

        cleanup()

        return attn_head_at_stage, terminal_heads_at_stage
    elif relax_mode == "dag":
        # DAG relax
        terminal_threshold = 0.5
        branching_factor = 50
        max_nodes_relax = 500

        canonical_heads = {}

        def canonize(attn_head, score, ret):
            canonical_heads[attn_head] = {
                "score": score,
                "children": [(d["attn_head"], d["score"]) for d in ret],
            }

        head_queue = DoubleQ()
        for relax_count in trange(max_nodes_relax):
            if relax_count == 0:
                ret = relax_node(attn_head=None, vec=None, is_root=True)
                canonize(attn_head=None, ret=ret, score=None)
                for d in ret:
                    head_queue.insert(d["attn_head"], d)
            else:
                if not head_queue.data:
                    break
                head, head_deets = head_queue.pop_topo()
                is_terminal, direct_score = check_terminal(head, head_deets["vec"])
                direct_child = [{"attn_head": None, "score": direct_score}]
                if is_terminal:
                    canonize(
                        attn_head=head, ret=direct_child, score=head_deets["score"]
                    )
                else:
                    ret = relax_node(
                        attn_head=head, vec=head_deets["vec"], is_root=False
                    )
                    canonize(
                        attn_head=head,
                        ret=ret + direct_child,
                        score=head_deets["score"],
                    )
                    for d in ret:
                        head_queue.insert(d["attn_head"], d)
        cleanup()

        return canonical_heads


def validate_terminal(
    input_tokens,
    prompt_ids,
    stacked_token_maps,
    answer_tokens,
    category,
    terminal_heads,
):
    hook_context = {}
    _, exogenous_cache = get_exogenous_cache(
        input_tokens=input_tokens,
        prompt_ids=prompt_ids,
        stacked_token_maps=stacked_token_maps,
        answer_tokens=answer_tokens,
        category=category,
    )
    # _, normal_cache = get_normal_cache(input_tokens=input_tokens, prompt_ids=prompt_ids, stacked_token_maps=stacked_token_maps, answer_tokens=answer_tokens)
    # logits, abnormal_cache = get_abnormal_cache(input_tokens=input_tokens, prompt_ids=prompt_ids, stacked_token_maps=stacked_token_maps, answer_tokens=answer_tokens, hook_context=hook_context, category=category)

    forward_name = lambda layer: f"blocks.{layer}.attn.hook_z"
    backward_name = lambda layer: f"blocks.{layer}.attn.hook_z_grad"

    # now, run forward pass, but patch over terminal_heads
    layer_dict = {layer: [] for layer, head, pos in terminal_heads}
    for layer, head, pos in terminal_heads:
        layer_dict[layer].append((pos, head))

    def patch_head_pos(acts, hook, pos_head_list):
        # acts: batch, pos, head_idx, dim
        for pos, head in pos_head_list:
            acts[:, pos, head, :] = exogenous_cache[hook.name][:, pos, head, :]
        return acts

    terminal_head_hooks = [
        (forward_name(layer), partial(patch_head_pos, pos_head_list=pos_head_list))
        for layer, pos_head_list in layer_dict.items()
    ]
    with _model.hooks(fwd_hooks=terminal_head_hooks):
        logits = normal_run(
            input_tokens=input_tokens,
            prompt_ids=prompt_ids,
            stacked_token_maps=stacked_token_maps,
            answer_tokens=answer_tokens,
        )
    return logits


def patch_queries(
    input_tokens,
    prompt_ids,
    stacked_token_maps,
    answer_tokens,
    category,
    terminal_heads,
):
    hook_context = {}

    logits, abnormal_cache = cint.run_cached_model(  # [batch, pos, head_index, d_head]
        execution_fn=lambda: swap_category_run(
            input_tokens=input_tokens,
            prompt_ids=prompt_ids,
            stacked_token_maps=stacked_token_maps,
            answer_tokens=answer_tokens,
            category=category,
        ),
        model=_model,
        names_filter=lambda x: "hook_q" in x,
    )

    # _, normal_cache = get_normal_cache(input_tokens=input_tokens, prompt_ids=prompt_ids, stacked_token_maps=stacked_token_maps, answer_tokens=answer_tokens)
    # logits, abnormal_cache = get_abnormal_cache(input_tokens=input_tokens, prompt_ids=prompt_ids, stacked_token_maps=stacked_token_maps, answer_tokens=answer_tokens, hook_context=hook_context, category=category)

    forward_name = lambda layer: f"blocks.{layer}.attn.hook_q"

    # now, run forward pass, but patch over terminal_heads
    layer_dict = {layer: [] for layer, head, pos in terminal_heads}
    for layer, head, pos in terminal_heads:
        layer_dict[layer].append((pos, head))

    def patch_head_pos(acts, hook, pos_head_list):
        # acts: batch, pos, head_idx, dim
        for pos, head in pos_head_list:
            acts[:, pos, head, :] = abnormal_cache[hook.name][:, pos, head, :]
        return acts

    terminal_head_hooks = [
        (forward_name(layer), partial(patch_head_pos, pos_head_list=pos_head_list))
        for layer, pos_head_list in layer_dict.items()
    ]
    with _model.hooks(fwd_hooks=terminal_head_hooks):
        logits = normal_run(
            input_tokens=input_tokens,
            prompt_ids=prompt_ids,
            stacked_token_maps=stacked_token_maps,
            answer_tokens=answer_tokens,
        )
    return logits


def my_max(ls, cmp):
    r = ls[0]
    for i in ls:
        if cmp(r, i):
            r = i
    return r


class DoubleQ:
    @staticmethod
    def topo_cmp(head1, head2):
        if head1[0] != head2[0]:
            return head1[0] < head2[0]  # layer
        if head1[2] != head2[2]:
            return head1[2] < head2[2]  # pos
        return head1[1] < head2[1]  # head_idx

    def __init__(self, hard_cap=50):
        self.hard_cap = hard_cap
        self.data = {}

    def insert(self, head, d):
        if head in self.data:
            self.data[head]["score"] += d["score"]
            self.data[head]["vec"] += d["vec"]
        else:
            self.data[head] = {
                "score": d["score"],
                "vec": d["vec"].clone(),
            }

    def pop_topo(self):
        """
        Among the top `hard_cap` heads, find the topologically last guy, and
        additionally delete everything that is topologically behind this guy
        """
        scores = sorted(
            [(head, d["score"]) for head, d in self.data.items()], key=lambda x: -x[1]
        )[: self.hard_cap]
        ret = my_max([head for head, _ in scores], cmp=self.topo_cmp)
        for head in list(self.data.keys()):
            if self.topo_cmp(ret, head):
                del self.data[head]
        return ret, self.data.pop(ret)
