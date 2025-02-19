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

from coref.datasets.templates.fixed import NameCountryFoodFixedTemplate, NameCountryFoodOccupationFixedTemplate, NameCountryOccupationFixedTemplate

import coref.form_processors
import coref.probes.lookup_probe as lkp

def compute_auroc(positive_scores, negative_scores):

    from sklearn.metrics import roc_auc_score
    return roc_auc_score(
        y_true=[1] * len(positive_scores) + [0] * len(negative_scores), 
        y_score=positive_scores + negative_scores
    )
def compute_auprc(positive_scores, negative_scores):

    from sklearn.metrics import average_precision_score
    return average_precision_score(
        y_true=[1] * len(positive_scores) + [0] * len(negative_scores), 
        y_score=positive_scores + negative_scores
    )
def recompute_domain_values(
    mask,
    scores,
    probe,
    threshold
):
    probe_results = centroid_probe.score_to_prediction(
        scores=scores,
        threshold=threshold
    )
    new_values = centroid_probe.results_to_values(
        mask=mask,
        probe_results=probe_results, 
        probe=probe
    )
    return new_values

def cat_with_pad(tensors, dim, pad, pad_dim):
    max_len = max(t.shape[pad_dim] for t in tensors)
    padded_tensors = []
    for t in tensors:
        pad_shape = list(t.shape)
        pad_shape[pad_dim] = max_len - t.shape[pad_dim]
        pad_tensor = torch.full(pad_shape, pad, dtype=t.dtype, device=t.device)
        padded_tensors.append(torch.cat([t, pad_tensor], dim=pad_dim))
    return torch.cat(padded_tensors, dim=dim)

def compute_metrics(all_true_values, all_scores, all_masks, probe):
    # all_true_values: list (over batches) of list of (value, type)
    # all_scores: [value, batch, pos]
    # all_masks: [batch, pos]
    max_scores = all_scores.max(dim=-1).values
    
    positive_scores = []
    negative_scores = []
    for batch, true_values in enumerate(all_true_values):
        true_values = set(value for value, _ in true_values)
        for value_idx, value in enumerate(probe.values):
            if value in true_values:
                positive_scores.append(max_scores[value_idx, batch].item())
            else:
                negative_scores.append(max_scores[value_idx, batch].item())
    auroc = compute_auroc(positive_scores, negative_scores)
    auprc = compute_auprc(positive_scores, negative_scores)

    hard_threshold = (np.min(positive_scores) + np.max(negative_scores)) / 2
    soft_threshold = (np.quantile(positive_scores, 0.25) + np.quantile(negative_scores, 0.9)) / 2
    def get_accuracy(threshold):
        all_predicted_values = recompute_domain_values(
            mask=all_masks,
            scores=all_scores,
            probe=probe,
            threshold=threshold
        )
        return np.mean([set(true_predicates) == set(predicates) for true_predicates, predicates in zip(all_true_values, all_predicted_values)])
    return {
        'auroc': auroc,
        'auprc': auprc,
        'margin': np.min(positive_scores) - np.max(negative_scores),
        'soft_margin': np.quantile(positive_scores, 0.25) - np.quantile(negative_scores, 0.9),
        'hard_threshold': hard_threshold,
        'soft_threshold': soft_threshold,
        'threshold_acc': get_accuracy(hard_threshold),
        'soft_threshold_acc': get_accuracy(soft_threshold)
    }
def evaluate_domain_probe_sweep_layers(
    model,
    dataset,
    domain_decoder,
    batch_size,
    unique_type='NAMES',
):
    all_scores = []
    all_masks = []
    for c_slice in eu.generate_slices(0, len(dataset), batch_size):
        scores, mask = domain_decoder.sweep_layers(
            prefixes=[d['prefix'] for d in dataset[c_slice]],
            contexts=[d['context'] for d in dataset[c_slice]],
            model=model,
        )
        # scores: [layer, values, batch, pos]
        # mask : [batch, pos]
        all_scores.append(scores)
        all_masks.append(mask)
    all_scores = cat_with_pad(all_scores, dim=2, pad=-torch.inf, pad_dim=3) #  layer, values, batch, pos
    all_masks = cat_with_pad(all_masks, dim=0, pad=0, pad_dim=1) # batch, pos
    if domain_decoder.probe.type != unique_type:
        all_true_values = [
            [
                (value, type)
                for name, value, type in d['predicates'] if type == domain_decoder.probe.type
            ]
            for d in dataset
        ]
    else:
        all_true_values = [
            [
                (name, unique_type)
                for name, value, type in d['predicates']
            ]
            for d in dataset
        ]
    metrics = [
        compute_metrics(all_true_values, all_scores[layer], all_masks, domain_decoder.probe)
        for layer in range(model.cfg.n_layers)
    ]
    data = {
        'all_true_values': all_true_values,
        'all_masks': all_masks.cpu(),
        'all_scores': all_scores.cpu(),
    }
    return metrics, data
def evaluate_domain_probe(
    model,
    dataset,
    domain_decoder,
    batch_size,
    unique_type='NAMES',
):
    if domain_decoder.probe.threshold is None:
        logging.warn(f'Probe {domain_decoder.probe} has no threshold set, using default threshold=6')
        domain_decoder.probe.threshold = 6

    all_predicted_values = []
    all_scores = []
    all_masks = []
    max_scores = []
    for c_slice in eu.generate_slices(0, len(dataset), batch_size):
        predicted_predicates, scores, mask = domain_decoder(
            prefixes=[d['prefix'] for d in dataset[c_slice]],
            contexts=[d['context'] for d in dataset[c_slice]],
            model=model,
            output_scores=True
        )
        all_predicted_values.extend(predicted_predicates)
        max_scores.append(torch.where(mask.to(scores.device), scores, -torch.inf).max(dim=-1).values)
        all_scores.append(scores)
        all_masks.append(mask)
    all_scores = cat_with_pad(all_scores, dim=1, pad=-torch.inf, pad_dim=2) #  value, batch, pos
    max_scores = torch.cat(max_scores, dim=1) #  value, batch
    all_masks = cat_with_pad(all_masks, dim=0, pad=0, pad_dim=1) # batch, pos
    if domain_decoder.probe.type != unique_type:
        all_true_values = [
            [
                (value, type)
                for name, value, type in d['predicates'] if type == domain_decoder.probe.type
            ]
            for d in dataset
        ]
    else:
        all_true_values = [
            [
                (name, unique_type)
                for name, value, type in d['predicates']
            ]
            for d in dataset
        ]
    acc = [set(true_predicates) == set(predicates) for true_predicates, predicates in zip(all_true_values, all_predicted_values)]
    
    metrics = compute_metrics(all_true_values, all_scores, all_masks, domain_decoder.probe)
    metrics.update(acc=np.mean(acc))

    data = {
        'all_true_values': all_true_values,
        'all_masks': all_masks.cpu(),
        'all_scores': all_scores.cpu(),
    }
    return metrics, data

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
    local_dir: Optional[str] = None

    chat_style: Literal['llama_chat', 'tulu_chat'] = 'llama_chat'

    has_occupation : bool = False
    has_food : bool = True
    use_cached_probes: bool = False
    probe_cache_dir: Optional[str] = None
    probe_target_layer: int = 20

    evaluate_domain_probes: bool = True # deprecated, does nothing now

    form_path: Optional[str] = None
    form_type: Optional[str] = None
    das_path: Optional[str] = None

    dataset_path: Optional[str] = None

    affinity_fn: str = "U_subspace_affinity_fn"
    affinity_fn_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            'dim': 50,
            'layer': 15
        }
    )

    probe_type: Literal['prompt', 'lookup'] = 'lookup'
    probe_enforce_matching: bool = False

    prefix_overwrite: Optional[str] = None


f'''
python -m coref.probes.evaluate \
    --config experiments/probes/namecountryfood_fixed.yaml
'''

@rm.automain
def main(cfg, output_dir):
    logging.output_dir = output_dir
    cfg = Cfg(**cfg)
    
    if cfg.has_occupation and cfg.has_food:
        fixed_template = NameCountryFoodOccupationFixedTemplate('llama')
    elif cfg.has_food:
        fixed_template = NameCountryFoodFixedTemplate('llama')
    elif cfg.has_occupation:
        fixed_template = NameCountryOccupationFixedTemplate('llama')
    else:
        if cfg.dataset_path is None:
            logging.error("Must have either occupation or food")
            assert False, "Must have either occupation or food"
        else:
            # we only need fixed_template for access to the domains
            fixed_template = NameCountryFoodFixedTemplate('llama')

    model = models.fetch_model(cfg.model, num_devices=cfg.num_devices, dtype=torch.bfloat16, hf=cfg.is_hf, peft_dir=cfg.peft_dir, local_dir=cfg.local_dir)    


    if cfg.dataset_path is None:
        logging.info("dataset_path not supplied. Evaluating on fixed template by default")
        dataset = lkp.template_to_dataset(
            template=fixed_template,
            prompt_type='chat_name_country',
            num_entities=2,
            num_samples=128,
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
    if cfg.prefix_overwrite is not None:
        for d in dataset:
            d['prefix'] = cfg.prefix_overwrite
    if cfg.probe_type == 'lookup':
        cfg.probe_cache_dir = Path(cfg.probe_cache_dir)
        predicate_probe = lkp.compose_lookup_probe(
            model=model,
            use_cached_probes=cfg.use_cached_probes,
            probe_target_layer=cfg.probe_target_layer,
            form_path=cfg.form_path,
            das_path=cfg.das_path,
            form_type=cfg.form_type,
            affinity_fn=cfg.affinity_fn,
            affinity_fn_kwargs=cfg.affinity_fn_kwargs,
            probe_cache_dir=cfg.probe_cache_dir,
            has_country=True,
            has_food=cfg.has_food,
            has_occupation=cfg.has_occupation,
            chat_style=cfg.chat_style,
            enforce_matching=cfg.probe_enforce_matching
        )
        decoder = lkp.PredicateProbeDecoder(predicate_probe, chat_style=cfg.chat_style)
    else:
        decoder = lkp.PromptDecoder(
            name_domain=fixed_template.names,
            country_domain=fixed_template.capitals,
            food_domain=fixed_template.foods if cfg.has_food else None,
            occupation_domain=fixed_template.occupations if cfg.has_occupation else None,
            chat_style=cfg.chat_style
        )
    # eval everything
    logging.info(f'Evaluating predicate probe')
    acc, all_predicates, all_true_predicates = lkp.evaluate_predicate_probe(
        model=model,
        dataset=dataset,
        decoder=decoder,
        num_samples=len(dataset),
        batch_size=32,
    )
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({
            'acc': acc,
        }, f)
    logging.info(f'Accuracy: {acc}')
    with open(output_dir / "details.json", "w") as f:
        json.dump({
            'all_predicates': all_predicates,
            'all_true_predicates': all_true_predicates
        }, f)