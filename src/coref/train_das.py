
from tqdm import trange
from importlib import reload
import os
from pathlib import Path
import json
from coref import COREF_ROOT
from coref.utils import cache
from functools import partial
import itertools

import torch
import torch.nn as nn
import numpy as np
import einops
from transformers import get_linear_schedule_with_warmup


import coref.models as models
import coref.experiments.triplet_expts as te
import coref.datasets.templates.triplet as tt


import coref.datasets.templates.simple as ts
import coref.datasets.templates.common as tc
import coref.datasets.templates.triplet as tt
import coref.datasets.api as ta
import coref.parameters as p
import coref.datascience as ds
import coref.expt_utils as eu
import coref.injection_interventions as ii
import coref.eval_subspace as ess
import coref.form_processors

from exults.log_utils import Logger
import logging

class Subspace(nn.Module):
    def __init__(self, d_model, d_subspace, dtype, device):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.empty((d_model, d_subspace), dtype=dtype, device=device))
        nn.init.orthogonal_(self.weight)
        nn.utils.parametrizations.orthogonal(self)

@torch.enable_grad()
def train_das_subspace(d_subspace, model, train_template, test_template, category, chat_style, logger):
    # follow procedure in pyvene
    # https://github.com/stanfordnlp/pyvene/blob/ffad51a169d0dc76f020da0d62fc14ee82732746/pyvene/models/intervenable_base.py#L1730
    lr = 1e-3
    num_samples = 128
    num_test_samples = 128
    epochs = 5
    warm_up_steps_ratio = 0.1
    batch_size = 8
    
    
    t_total = (num_samples // batch_size) * epochs
    warm_up_steps = warm_up_steps_ratio * t_total
        
    for param in model.parameters():
        param.requires_grad = False
    
    subspace = Subspace(model.cfg.d_model, d_subspace, torch.float32, 'cuda')
    
    optimizer = torch.optim.Adam(subspace.parameters(), lr=lr, maximize=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
    )
    def eval_loss(template, batch_size, prompt_id_start, return_acc=False):
        res = ess.eval_project_switch(
            model=model,
            subspace=subspace.weight.to(torch.bfloat16),
            category=category,
            chat_style=chat_style,
            test_template=template,
            side1=0,
            side2=1,
            num_entities=2,
            num_samples=batch_size,
            batch_size=batch_size,
            prompt_id_start = prompt_id_start
        )
        loss = eu.reduce_mean_correct_logit(res, [1, 0]).mean()
        if return_acc:
            acc = eu.reduce_acc(res, [1, 0]).mean()
            return loss, acc
        else:
            return loss
    @torch.no_grad()
    def get_test_loss():
        loss_sum = 0
        acc_sum = 0
        for prompt_id in range(0, num_test_samples, batch_size):
            loss, acc = eval_loss(
                template=test_template,
                batch_size=batch_size,
                prompt_id_start=prompt_id,
                return_acc=True
            )
            loss_sum += loss.item() * batch_size
            acc_sum += acc.item() * batch_size
        return loss_sum / num_test_samples, acc_sum / num_test_samples
    
    logger.reset()
    logger.init('lrs')
    logger.init('train_losses')
    logger.init('test_losses')
    logger.init('test_accs')
        
    test_loss, test_acc = get_test_loss()
    logger.test_losses.append(test_loss)
    logger.test_accs.append(test_acc)
    
    for epoch in trange(0, epochs):
        for prompt_id in range(0, num_samples, batch_size):
            loss = eval_loss(
                template=train_template,
                batch_size=batch_size,
                prompt_id_start=prompt_id
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            logger.train_losses.append(loss.item())
            logger.lrs.append(scheduler.get_last_lr()[0])
        test_loss, test_acc = get_test_loss()
        logger.test_losses.append(test_loss)
        logger.test_accs.append(test_acc)
        
    return subspace.weight


import coref.run_manager as rm
import json

from dataclasses import dataclass, field
from typing import Literal, Any, Optional, Dict
@dataclass(kw_only=True)
class Cfg(rm.Cfg):
    model: str
    num_devices: int = 1
    is_hf: bool = False

    d_subspace: int
    
    template: str = "NameCountryTemplate"
    chat_style: Literal['llama_chat', 'tulu_chat'] = 'llama_chat'

@rm.automain
def main(cfg, output_dir):
    logging.output_dir = output_dir
    cfg = Cfg(**cfg)

    model = models.fetch_model(cfg.model, num_devices=cfg.num_devices, dtype=torch.bfloat16, hf=cfg.is_hf)


    logger = Logger(output_dir / 'logs' / 'name_subspace')
    subspace = train_das_subspace(
        d_subspace=cfg.d_subspace,
        model=model,
        train_template=ta.get_template(cfg.template)('llama', 'train'),
        test_template=ta.get_template(cfg.template)('llama', 'test'),
        category='name',
        chat_style=cfg.chat_style,
        logger=logger
    )
    torch.save(subspace, output_dir / 'name_subspace.pt')
    logger = Logger(output_dir / 'logs' / 'attr_subspace')
    subspace = train_das_subspace(
        d_subspace=cfg.d_subspace,
        model=model,
        train_template=ta.get_template(cfg.template)('llama', 'train'),
        test_template=ta.get_template(cfg.template)('llama', 'test'),
        category='attr',
        chat_style=cfg.chat_style,
        logger=logger
    )
    torch.save(subspace, output_dir / 'attr_subspace.pt')


    # evaluation
    (output_dir/'eval').mkdir(exist_ok=True)

    form = coref.form_processors.das_form(output_dir)
    test_template = ta.get_template(cfg.template)('llama', 'test')
    # name
    metrics, data = ess.eval_subspace(
        model=model,
        subspace=form.U.to(torch.bfloat16),
        category='name',
        chat_style=cfg.chat_style,
        test_template=test_template,
        verbose=False
    )
    with open(output_dir / 'eval/name_metrics.json', 'w') as f:
        json.dump(metrics, f)
    # attr
    metrics, data = ess.eval_subspace(
        model=model,
        subspace=form.Vh.T.to(torch.bfloat16),
        category='attr',
        chat_style=cfg.chat_style,
        test_template=test_template,
        verbose=False
    )
    with open(output_dir / 'eval/attr_metrics.json', 'w') as f:
        json.dump(metrics, f)
