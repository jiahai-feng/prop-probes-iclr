from functools import partial
import numpy as np
import torch
import einops

import coref.datasets.templates.simple as ts
import coref.datasets.templates.common as tc
import coref.datasets.templates.triplet as tt
import coref.datasets.api as ta
import coref.parameters as p
import coref.datascience as ds
import coref.expt_utils as eu

attr_name_map = {0: "attr0", 1: "attr1", 2: "attr2"}


def build_ctx(ctx_values):
    return [tc.TStatement(*r, None) for r in ctx_values]


def mean_interventions(
    *,
    model,
    prompt_id_start,
    const_attr,
    vary_attr,
    ans_attr,
    train_template,
    test_template,
    category_widths=dict(attr0=2, attr1=2, attr2=2),
    num_mean_samples=500,
    num_samples=100,
):
    def build_query(const_attr_val, vary_attr_val):
        if const_attr < vary_attr:
            attrA, attrB = const_attr_val, vary_attr_val
        else:
            attrA, attrB = vary_attr_val, const_attr_val
        return dict(attrA=attrA, attrB=attrB, prompt_type=attr_name_map[ans_attr])

    target_ctx_values = np.zeros((2, 3), dtype=int)
    target_ctx_values[:, const_attr] = 0
    target_ctx_values[:, vary_attr] = np.array([0, 1])
    target_ctx_values[:, ans_attr] = np.array([0, 1])
    target_ctx = build_ctx(target_ctx_values)

    both_acts = [
        ds.collect_data(
            model=model,
            num_samples=num_mean_samples,
            batch_size=20,
            template=train_template,
            template_content=dict(attrA=0, attrB=0, prompt_type="attr2"),
            context_content=ctx,
            num_answers=2,
            prompt_id_start=prompt_id_start,
            extractor_map=lambda token_maps: {
                "attr0": [
                    [x + i for i in range(category_widths["attr0"])]
                    for x in token_maps["attr0"]
                ],
                "attr1": [
                    [x + i for i in range(category_widths["attr1"])]
                    for x in token_maps["attr1"]
                ],
                "attr2": [
                    [x + i for i in range(category_widths["attr2"])]
                    for x in token_maps["attr2"]
                ],
            },
        )[
            0
        ]  # returns acts, tokens, ans by default
        for ctx in [
            [
                tc.TStatement(0, 0, 0, None),
                tc.TStatement(1, 1, 1, None),
            ],
            [
                tc.TStatement(1, 1, 1, None),
                tc.TStatement(0, 0, 0, None),
            ],
        ]
    ]
    means = {
        attr: p.build_mean_param(model, both_acts[0][attr][1], both_acts[1][attr][0])
        for attr in both_acts[0]
    }  #  batch layer dim

    const_attr_val = 0

    def get_swap_vars(swap_entity, swap_attr):
        ret = []
        if swap_entity:
            ret += [vary_attr]
        if swap_attr:
            ret += [ans_attr]
        return ret

    return eu.repeat(
        2,
        lambda swap_entity: eu.repeat(
            2,
            lambda swap_attr: eu.repeat(
                2,
                lambda vary_attr_val: p.evaluate_interventions(
                    model=model,
                    num_samples=num_samples,
                    batch_size=20,
                    template=test_template,
                    template_content=build_query(const_attr_val, vary_attr_val),
                    context_content=target_ctx,
                    num_answers=2,
                    prompt_id_start=prompt_id_start,
                    interventions=p.compose_interventions(
                        *[
                            p.get_inject_intervention(
                                means,
                                attr_name_map[attr],
                                target_side,
                                category_widths,
                                model,
                            )
                            for target_side in [0, 1]
                            for attr in get_swap_vars(swap_entity, swap_attr)
                        ]
                    ),
                    return_raw=True,
                ),
            ),
        ),
    )
