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
import torch
import einops
import logging

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

    # finite hessian
    form_path: str
    form_type: str

@rm.automain
def main(cfg, output_dir):
    cfg = Cfg(**cfg)
    base_notebook_path = os.path.join(COREF_ROOT, 'notebooks/eval_form_master.ipynb')
    papermill.execute_notebook(
        base_notebook_path,
        os.path.join(output_dir, "eval_form.ipynb"),
        parameters=dict(
            MODEL_TAG=cfg.model,
            NUM_DEVICES=cfg.num_devices,
            IS_HF=cfg.is_hf,
            template_tag=cfg.template,
            chat_style=cfg.chat_style,
            form_path=cfg.form_path,
            form_type=cfg.form_type,
            output_dir=str(output_dir)
        )
    )