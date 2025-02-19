from importlib import reload
import os
from coref import COREF_ROOT
from coref.utils import cache
from functools import partial
import itertools

import torch

import coref.models as models
import coref.experiments.triplet_expts as te
import coref.datasets.templates.triplet as tt

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

import coref.injection_interventions as ii

import coref.flow as flow

from tqdm import trange


@flow.flow
def compute_scales(
    model,
    train_template,
    prompt_type,
    train_prompt_id_start=0,
    category_widths={"name": 2, "attr": 1},
):
    num_mean_samples = 100
    with torch.no_grad():
        both_acts = [
            ds.collect_data(
                model=model,
                num_samples=num_mean_samples,
                batch_size=20,
                template=train_template,
                template_content=dict(query_name=0, chat_style=prompt_type),
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

    def mean_scale(t):
        # returns [layer, 1]
        return t.norm(dim=-1, keepdim=True).mean(dim=0)

    mean_scales = tc.recursify(
        mean_scale,
        dtype=torch.Tensor,
    )(both_acts)
    return dict(mean_scales=mean_scales, both_acts=both_acts)


@flow.flow
def prepare_fixed_interventions(
    model, both_acts, category_widths, interpolating_factor=0.5
):
    # interpolating_factor = 0 means binding id=0, 0.5 means midpoint between 0 and 1
    with torch.no_grad():
        means = {
            attr: p.build_mean_param(
                model, both_acts[0][attr][0], both_acts[1][attr][1]
            )
            for attr in both_acts[0]
        }  #  batch layer dim
        # this gives b(0) - b(1)
        mid_point_fixed_injections = [
            ii.injection_intervention_from_param(
                model=model,
                param=means[category],
                width=category_widths[category],
                category=category,
                target_side=side,
                adapter=ii.scale_adapter(
                    means[category].default_adapter,
                    (1 - interpolating_factor) if side else -interpolating_factor,
                ),
            )
            for category in ["name", "attr"]
            for side in [0, 1]
        ]
    return dict(mid_point_fixed_injections=mid_point_fixed_injections)


def scaled_f_mid_point(
    mean_scales,
    model,
    category_widths,
    test_template,
    mid_point_fixed_injections,
    prompt_type,
    x,
    y,
    prompt_id_start,
    num_samples,
    uniform_scale=False,
):
    if uniform_scale: # ablation
        uniform_x_scale = np.mean([
            mean_scales[0]["name"][0][0][layer, 0].item()
            for layer in range(model.cfg.n_layers)
        ])
        uniform_y_scale = np.mean([
            mean_scales[0]["attr"][0][0][layer, 0].item()
            for layer in range(model.cfg.n_layers)
        ])
    vary_params = {
        "name": p.ScaleParameters(
            parameters=x,
            scales={
                layer: np.mean(
                    [
                        mean_scales[0]["name"][0][0][layer, 0].item(),
                        mean_scales[0]["name"][0][1][layer, 0].item(),
                    ]
                ) if not uniform_scale else uniform_x_scale
                for layer in range(model.cfg.n_layers)
            },
        ),
        "attr": p.ScaleParameters(
            parameters=y,
            scales={
                layer: np.mean(
                    [
                        mean_scales[0]["attr"][0][0][layer, 0].item(),
                    ]
                ) if not uniform_scale else uniform_y_scale
                for layer in range(model.cfg.n_layers)
            },
        ),
    }

    vary_injections = [
        ii.injection_intervention_from_param(
            model=model,
            param=vary_params[category],
            width=category_widths[category],
            category=category,
            target_side=1,
            adapter=vary_params[category].default_adapter,
        )
        for category in ["name", "attr"]
    ]
    target_ctx = [
        tc.Statement(0, 0, None),
        tc.Statement(1, 1, None),
    ]

    v1 = [
        p.evaluate_interventions(
            model=model,
            num_samples=num_samples,
            batch_size=num_samples,
            template=test_template,
            template_content=dict(query_name=query_name, chat_style=prompt_type),
            context_content=target_ctx,
            num_answers=2,
            prompt_id_start=prompt_id_start,
            interventions=p.compose_interventions(
                *(mid_point_fixed_injections + vary_injections)
            ),
            return_raw=False,
        )
        for query_name in range(2)
    ]

    probs = 0.5 * (torch.exp(v1[0][0]) + torch.exp(v1[1][1]))
    return probs


prepare_stuff = flow.stream([compute_scales, prepare_fixed_interventions])


def evaluate_f(*, namespace, **kwargs):
    return flow.lift_start(scaled_f_mid_point)({**namespace, **kwargs})


def finite_hessian(
    func, model, widths=(1, 1), rms_vals=(2e-3, 2e-3), n_samples=4096, balance=False
):
    for param in model.parameters():
        param.requires_grad = False
    torch.set_grad_enabled(True)
    x_width, y_width = widths
    x_rms, y_rms = rms_vals
    xpes = []
    yes = []
    xes = []
    ypes = []
    for i in trange(n_samples):
        if balance and i % 2:
            x = -xes[-1].clone()
            y = -yes[-1].clone()
        x = torch.normal(0, x_rms, size=(x_width, model.cfg.d_model))
        y = torch.normal(0, y_rms, size=(y_width, model.cfg.d_model))
        x.requires_grad = True
        y.requires_grad = True
        first = func(x, y)
        first.backward()
        xpes.append(x.grad.clone().detach().cpu())
        yes.append(y.clone().detach().cpu())
        xes.append(x.clone().detach().cpu())
        ypes.append(y.grad.clone().detach().cpu())

    return torch.stack(xes), torch.stack(yes), torch.stack(xpes), torch.stack(ypes)


def point_hessian(func, model, widths=(1, 1), swap_dir=False):
    for param in model.parameters():
        param.requires_grad = False
    torch.set_grad_enabled(True)
    if swap_dir:
        widths = widths[::-1]
        real_func = lambda x, y: func(y, x)
    else:
        real_func = func
    x_width, y_width = widths
    x = torch.zeros((x_width, model.cfg.d_model), requires_grad=True)
    y = torch.zeros((y_width, model.cfg.d_model), requires_grad=True)
    first = real_func(x, y)
    first.backward(create_graph=True)
    x.requires_grad = True
    y.requires_grad = False
    y_grad = y.grad.clone()
    hessian = torch.zeros((x_width, model.cfg.d_model, y_width, model.cfg.d_model))
    for w in trange(y_width):
        for i in trange(model.cfg.d_model):
            x.grad = None
            y_grad[w, i].backward(retain_graph=True)
            hessian[:, :, w, i] = x.grad.clone().detach()
    if swap_dir:
        hessian = einops.rearrange(hessian, "yw y xw x -> xw x yw y")
    return hessian
