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

from tqdm import trange

import coref.eval_subspace as ess

import json

@torch.no_grad()
def get_binding_vectors(
    model,
    train_template,
    prompt_type,
    train_prompt_id_start=0,
    category_widths={"name": 2, "attr": 1},
):
    num_mean_samples = 100
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
                tc.Statement(2, 2, None),
            ],
            [
                tc.Statement(1, 1, None),
                tc.Statement(2, 2, None),
                tc.Statement(0, 0, None),
            ],
        ]
    ]

    means_0 = { # b(1) - b(0)
        attr: p.build_mean_param(
            model, both_acts[0][attr][1], both_acts[1][attr][0],
        )
        for attr in category_widths
    }  

    means_1 = { # b(2) - b(0)
        attr: p.build_mean_param(
            model, both_acts[1][attr][2], both_acts[0][attr][0]
        )
        for attr in category_widths
    }  

    return dict(means_0=means_0, means_1=means_1)

def eval_subspace(model, chat_style, test_template, param_subspace, category):
    metrics = []
    data = []
    res = ess.eval_project_switch(
        model=model,
        category=category,
        chat_style=chat_style,
        test_template=test_template,
        side1=0,
        side2=1,
        num_entities=2,
        param_subspace=param_subspace
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
    res = ess.eval_project_switch(
        model=model,
        param_subspace=param_subspace,
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
    res = ess.eval_project_switch(
        model=model,
        param_subspace=param_subspace,
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
    return metrics, data

@torch.no_grad()
def evaluate(
    model,
    n_vecs,
    train_template,
    test_template,
    chat_style,
    output_dir
):
    
    output_dir.mkdir(exist_ok=True)
    assert n_vecs in [2, 3]
    binding_vectors = get_binding_vectors(
        model=model,
        train_template=train_template,
        prompt_type=chat_style,
    )
    def orthogonalize_mean_parameters(mpa, mpb):
        subspace_dict = {}
        for layer in mpa.parameters:
            assert len(mpa.parameters[layer].shape) == 2 and mpa.parameters[layer].shape[1] == model.cfg.d_model
            assert len(mpb.parameters[layer].shape) == 2 and mpb.parameters[layer].shape[1] == model.cfg.d_model
            vecs = torch.stack([
                mpa.parameters[layer].mean(dim=0),
                mpb.parameters[layer].mean(dim=0)
            ]).float()
            q, r = torch.linalg.qr(vecs.T, mode='reduced')
            subspace_dict[layer] = q.to(dtype=mpa.parameters[layer].dtype)

        return p.DictParameters.from_dict(model=model, d=subspace_dict)
    def normalize_mean_parameters(mpa):
        subspace_dict = {}
        for layer in mpa.parameters:
            assert len(mpa.parameters[layer].shape) == 2 and mpa.parameters[layer].shape[1] == model.cfg.d_model
            v = mpa.parameters[layer].mean(dim=0)
            v /= v.norm()
            subspace_dict[layer] = v[:, None]

        return p.DictParameters.from_dict(model=model, d=subspace_dict)
    
    for category in ['name', 'attr']:
        if n_vecs == 3:
            param_subspace = orthogonalize_mean_parameters(
                binding_vectors['means_0'][category],
                binding_vectors['means_1'][category],
            )
        else:
            param_subspace = normalize_mean_parameters(binding_vectors['means_0'][category])
        metrics, data = eval_subspace(
            model=model,
            chat_style=chat_style,
            param_subspace=param_subspace,
            test_template=test_template,
            category=category
        )
        with open(output_dir / f'{category}_metrics.json', 'w') as f:
            json.dump(metrics, f)
        torch.save(data, output_dir / f'{category}_data.pt')
        torch.save(param_subspace.parameters, output_dir / f'{category}_subspace.pt')


from dataclasses import dataclass
from typing import Literal, Any, Optional
import coref.run_manager as rm

@dataclass(kw_only=True)
class Cfg(rm.Cfg):
    model: str
    num_devices: int = 1
    is_hf: bool = False
    
    template: str = "NameCountryTemplate"
    chat_style: str = 'llama_chat'


@rm.automain
def main(cfg, output_dir):
    cfg = Cfg(**cfg)

    model = models.fetch_model(cfg.model, num_devices=cfg.num_devices, dtype=torch.bfloat16, hf=cfg.is_hf)

    train_template = ta.get_template(cfg.template)('llama')
    test_template = ta.get_template(cfg.template)('llama')


    evaluate(
        model=model,
        n_vecs=3,
        train_template=train_template,
        test_template=test_template,
        chat_style=cfg.chat_style,
        output_dir=output_dir/'3_vecs'
    )