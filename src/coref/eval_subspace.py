import torch
import coref.injection_interventions as ii
import coref.parameters as p
from functools import partial
import numpy as np
import torch
import einops
import torch.nn.functional as F

import coref.datasets.templates.simple as ts
import coref.datasets.templates.common as tc
import coref.datasets.templates.triplet as tt
import coref.datasets.api as ta
import coref.datascience as ds
import coref.expt_utils as eu


def orthogonalize(matrix):
    U = torch.linalg.svd(matrix, full_matrices=False)[0]
    return U


def project_injection_param(param, op):
    for k, x in param.parameters.items():
        param.parameters[k] = op(x)
    return param


def project(x, subspace):
    if len(subspace.shape) == 2:
        subspace = einops.repeat(subspace, "dim i -> w dim i", w=x.shape[0])
    return einops.einsum(
        x.float(),
        subspace.float(),
        subspace.float(),
        "w dim, w dim i, w dim2 i -> w dim2",
    ).to(dtype=x.dtype)


def eval_project_mean(
    *,
    model,
    subspace,
    category,
    prompt_type,
    train_template,
    test_template,
    num_entities,
    side1,
    side2,
    category_widths={"name": 2, "attr": 1}
):
    """
    subspace = None means control test
    """
    num_mean_samples = 100
    prompt_id_start = 0

    with torch.no_grad():
        both_acts = [
            ds.collect_data(
                model=model,
                num_samples=num_mean_samples,
                batch_size=20,
                template=train_template,
                template_content=dict(query_name=0, prompt_type=prompt_type),
                context_content=ctx,
                num_answers=num_entities,
                prompt_id_start=prompt_id_start,
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
                [tc.Statement(i, i, None) for i in range(num_entities)],
                [
                    tc.Statement((i + 1) % num_entities, (i + 1) % num_entities, None)
                    for i in range(num_entities)
                ],
            ]
        ]
        means = {
            attr: p.build_mean_param(
                model, both_acts[0][attr][side2], both_acts[1][attr][side1]
            )
            for attr in both_acts[0]
        }  #  batch layer dim
        if subspace is not None:
            project_injection_param(
                means[category], partial(project, subspace=subspace.cpu())
            )
    ints = ii.build_switch_interventions(
        model=model,
        category_widths=category_widths,
        param=means[category],
        category=category,
        target_sides=[side1, side2],
    )
    v1 = [
        p.evaluate_interventions(
            model=model,
            num_samples=100,
            batch_size=20,
            template=test_template,
            template_content=dict(query_name=query_name, prompt_type=prompt_type),
            context_content=[tc.Statement(i, i, None) for i in range(num_entities)],
            num_answers=num_entities,
            prompt_id_start=prompt_id_start,
            interventions=p.compose_interventions(*ints),
            return_raw=True,
        )
        for query_name in range(num_entities)
    ]

    return torch.stack(v1)


def eval_project_switch(
    *,
    model,
    subspace=None,
    category,
    test_template,
    side1,
    side2,
    num_entities,
    chat_style,
    param_subspace=None,
    category_widths={"name": 2, "attr": 1},
    num_samples=100,
    batch_size=20,
    prompt_id_start = 0
):
    '''
    Supply either subspace or param_subspace
    Args:
        subspace: torch.Tensor [d_model, subspace_dim] | None
        param_subspace: p.DictParameters | None
    '''

    v1 = [
        p.evaluate_interventions(
            model=model,
            num_samples=num_samples,
            batch_size=batch_size,
            template=test_template,
            template_content=dict(query_name=query_name, chat_style=chat_style),
            context_content=[tc.Statement(i, i, None) for i in range(num_entities)],
            num_answers=num_entities,
            prompt_id_start=prompt_id_start,
            interventions=p.compose_interventions(
                {
                    layer: lambda tensor, hook, prompt_ids, token_maps, layer=layer: p.project_switch_hook_fn(
                        tensor,
                        hook,
                        param=subspace if param_subspace is None else param_subspace.parameters[layer],
                        width=category_widths[category],
                        token_maps=token_maps,
                        extractor=lambda token_maps: (
                            token_maps[category][side1],
                            token_maps[category][side2],
                        ),
                    )
                    for layer in range(model.cfg.n_layers)
                }
            ),
            return_raw=True,
        )
        for query_name in range(num_entities)
    ]
    return torch.stack(v1)


def eval_subspace(subspace, category, model, chat_style, test_template, verbose=False):
    metrics = []
    data = []
    res = eval_project_switch(
        model=model,
        subspace=subspace,
        category=category,
        chat_style=chat_style,
        test_template=test_template,
        side1=0,
        side2=1,
        num_entities=2
    )
    metrics.append({
        'worst_acc': eu.reduce_balanced_acc(res.float(), [1, 0]).min().item(),
        'avg_acc': eu.reduce_balanced_acc(res.float(), [1, 0]).mean().item(),
        'switch': '0-1'
    })
    data.append({
        'switch': '0-1',
        'logits': res.cpu()
    })
    if verbose:
        torch.set_printoptions(profile='default', sci_mode=False)
        print('2 entity')
        print(eu.reduce_logit_mean(res))
        print(eu.reduce_balanced_acc(res.float(), [1, 0]))
        print(eu.reduce_acc(res.float(), [1, 0]))
    res = eval_project_switch(
        model=model,
        subspace=subspace,
        category=category,
        chat_style=chat_style,
        test_template=test_template,
        side1=0,
        side2=2,
        num_entities=3
    )
    metrics.append({
        'worst_acc': eu.reduce_balanced_acc(res.float(), [2, 1, 0]).min().item(),
        'avg_acc': eu.reduce_balanced_acc(res.float(), [2, 1, 0]).mean().item(),
        'switch': '0-2'
    })
    data.append({
        'switch': '0-2',
        'logits': res.cpu()
    })
    if verbose:
        print('0 <-> 2')
        print(eu.reduce_logit_mean(res))
        print(eu.reduce_balanced_acc(res.float(), [2, 1, 0]))
        print(eu.reduce_acc(res.float(), [2, 1, 0]))
    res = eval_project_switch(
        model=model,
        subspace=subspace,
        category=category,
        chat_style=chat_style,
        test_template=test_template,
        side1=1,
        side2=2,
        num_entities=3
    )
    metrics.append({
        'worst_acc': eu.reduce_balanced_acc(res.float(), [0, 2, 1]).min().item(),
        'avg_acc': eu.reduce_balanced_acc(res.float(), [0, 2, 1]).mean().item(),
        'switch': '1-2'
    })
    data.append({
        'switch': '1-2',
        'logits': res.cpu()
    })
    if verbose:
        print('1 <-> 2')
        print(eu.reduce_logit_mean(res))
        print(eu.reduce_balanced_acc(res.float(), [0, 2, 1]))
        print(eu.reduce_acc(res.float(), [0, 2, 1]))
        torch.set_printoptions(profile='default', sci_mode=None)
    return metrics, data