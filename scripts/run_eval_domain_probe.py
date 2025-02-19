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

    use_cached_probes: bool = False
    probe_cache_dir: Optional[str] = None
    probe_target_layer: int = 20


    use_class_conditioned: bool = True

    dataset_path: Optional[str] = None

    sweep_layers: bool = False




@rm.automain
def main(cfg, output_dir):
    logging.output_dir = output_dir
    cfg = Cfg(**cfg)
    assert cfg.probe_cache_dir is not None, "Must provide probe_cache_dir"
    cfg.probe_cache_dir = Path(cfg.probe_cache_dir)
    model = models.fetch_model(cfg.model, num_devices=cfg.num_devices, dtype=torch.bfloat16, hf=cfg.is_hf, peft_dir=cfg.peft_dir)
    if cfg.has_occupation:
        fixed_template = NameCountryFoodOccupationFixedTemplate('llama')
    else:
        fixed_template = NameCountryFoodFixedTemplate('llama')
        fixed_template.default_template_content['context_type'] = 'basic'
    


    if cfg.dataset_path is None:
        logging.info("dataset_path not supplied. Evaluating on basic template by default")
        dataset = lkp.template_to_dataset(
            template=fixed_template,
            prompt_type='chat_name_country',
            num_entities=2,
            num_samples=512,
            prompt_id_start=0
        )
    else:
        logging.info(f'Evaluating on dataset at {cfg.dataset_path}')
        # load and validate dataset
        with open(cfg.dataset_path, 'r') as f:
            dataset = json.load(f)
        for d in dataset:
            if not all(key in d for key in ['context', 'prompt_id', 'predicates']):
                raise ValueError('Dataset must have keys context, prompt_id, predicates')
            if 'prefix' not in d:
                d['prefix'] = ''
            d['predicates'] = lkp.list_to_tuple(d['predicates'])
    
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
        probe = probe_class.load_or_none(cfg.probe_cache_dir / f"{probe_name}.pt")
        assert probe is not None, (f"Could not load probes from {cfg.probe_cache_dir / f'{probe_name}.pt'}")
        probe.use_class_conditioned = cfg.use_class_conditioned
        
        eval_save_dir = output_dir / probe_name
        os.makedirs(eval_save_dir, exist_ok=True)
        logging.info(f'Evaluating domain probe {domain.type}')
        if cfg.sweep_layers:
            metrics, data = cpe.evaluate_domain_probe_sweep_layers(
                model=model,
                dataset=dataset,
                domain_decoder=lkp.DomainProbeDecoder(probe, chat_style=cfg.chat_style),
                batch_size=32
            )
        else:
            metrics, data = cpe.evaluate_domain_probe(
                model=model,
                dataset=dataset,
                domain_decoder=lkp.DomainProbeDecoder(probe, chat_style=cfg.chat_style),
                batch_size=32
            )
            logging.info(f'Accuracy: {metrics["acc"]}, AUROC: {metrics["auroc"]}')
        with open(eval_save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)

        torch.save(data, eval_save_dir / "details.pt")