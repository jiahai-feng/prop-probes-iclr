from functools import partial
import logging
import os
import numpy as np
import torch
import einops
import torch.nn.functional as F
from pathlib import Path

from coref import COREF_ROOT
import coref.models as models
import coref.parameters as p
import coref.probes.activation_localization as act_loc
import coref.probes.centroid_probe as centroid_probe
import coref.expt_utils as eu

from coref.datasets.domains.common import set_prompt_id
import coref.datasets.templates.common as tc

from coref.datasets.templates.fixed import NameCountryFoodFixedTemplate, NameCountryFoodOccupationFixedTemplate

import coref.form_processors
import coref.probes.lookup_probe as lkp
import coref.probes.evaluate as cpe


import coref.run_manager as rm
import json

from dataclasses import dataclass, field
from typing import Literal, Any, Optional, Dict
@dataclass(kw_only=True)
class Cfg(rm.Cfg):
    model: str
    num_devices: int = 1
    is_hf: bool = False
    peft_dir: Optional[str] = None

    chat_style: Literal['llama_chat', 'tulu_chat'] = 'llama_chat'

    has_occupation : bool = False

    probe_target_layer: int = 20
    sweep_layers: bool = False





@rm.automain
def main(cfg, output_dir):
    logging.output_dir = output_dir
    cfg = Cfg(**cfg)

    model = models.fetch_model(cfg.model, num_devices=cfg.num_devices, dtype=torch.bfloat16, hf=cfg.is_hf, peft_dir=cfg.peft_dir)
    if cfg.has_occupation:
        fixed_template = NameCountryFoodOccupationFixedTemplate('llama')
    else:
        fixed_template = NameCountryFoodFixedTemplate('llama')
        fixed_template.default_template_content['context_type'] = 'basic'
    
    target_layer = cfg.probe_target_layer if not cfg.sweep_layers else None

    probes_to_eval = [
        ("name_probe", centroid_probe.NameCentroidProbe, fixed_template.names),
        ("country_probe", centroid_probe.CountryCentroidProbe, fixed_template.capitals),
        ("food_probe", centroid_probe.FoodCentroidProbe, fixed_template.foods),
    ]
    if cfg.has_occupation:
        probes_to_eval.append(
            ("occupation_probe", centroid_probe.OccupationCentroidProbe, fixed_template.occupations)
        )

    for probe_name, probe_class, domain in probes_to_eval:
        probe = probe_class(target_layer=target_layer, chat_style=cfg.chat_style)
        probe.train(
            model=model,
            save_dir=output_dir / probe_name
        )
        probe.save(output_dir / f"{probe_name}.pt")