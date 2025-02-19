import papermill
from distutils.util import strtobool
from importlib import reload
import os
from coref import COREF_ROOT
from coref.utils import cache
from functools import partial
import itertools
import torch
import coref.hessians as ch
import coref.models as models
from functools import partial
import numpy as np
import einops
import logging
import coref.eval_subspace as ess
import coref.datasets.api as ta
import coref.form_processors
import json
from pathlib import Path
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

    das_path: str

@rm.automain
def main(cfg, output_dir):
    cfg = Cfg(**cfg)
    output_dir = Path(output_dir)
    model = models.fetch_model(cfg.model, num_devices=cfg.num_devices, dtype=torch.bfloat16, hf=cfg.is_hf)
    test_template = ta.get_template(cfg.template)('llama', 'test')

    form = coref.form_processors.das_form(Path(cfg.das_path))
    # name
    metrics, data = ess.eval_subspace(
        model=model,
        subspace=form.U.to(torch.bfloat16),
        category='name',
        chat_style=cfg.chat_style,
        test_template=test_template,
        verbose=False
    )
    with open(output_dir / 'name_metrics.json', 'w') as f:
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
    with open(output_dir / 'attr_metrics.json', 'w') as f:
        json.dump(metrics, f)

