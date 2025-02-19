import os
from coref import COREF_ROOT
from coref.utils import cache
from functools import partial
import itertools
import gc

import torch
import pandas as pd

import coref.models as models
import coref.experiments.triplet_expts as te
import coref.datasets.templates.triplet as tt

from functools import partial
import numpy as np
import torch
import einops
import torch.nn.functional as F

import coref.datasets.templates.simple as ts
import coref.datasets.templates.common as tc
import coref.datasets.templates.triplet as tt
import coref.datasets.domains.simple as simple_domains
from coref.datasets.domains.common import condition_domain
import coref.datasets.api as ta
import coref.parameters as p
import coref.datascience as ds
import coref.expt_utils as eu

import coref.interventions as cint
import coref.injection_interventions as ii

import coref.hf_interventions as hfint

import coref.flow as flow
import coref.eval_subspace as ess

import coref.saliency as sal
import coref.probes.activation_localization as act_loc
from coref.probes.common import DomainProbe

import logging


def localize_value(
    *,
    model, template, domain, num_entities, vary_entity, source_context, target_context, num_samples=100, batch_size=50, target_layer
):
    '''
    Return
        peak_position: int
        total_scores: FloatTensor[pos, layer]
    '''
    total_scores_acc = []
    for c_slice in eu.generate_slices(0, num_samples, batch_size):
        source_tokens, answer_tokens, stacked_token_maps = p.build_example(
            batch_size=c_slice.stop - c_slice.start,
            template=template,
            template_content=dict(query_name=vary_entity),
            context_content=source_context,
            prompt_id_start=c_slice.start,
            num_answers=num_entities*2,
        )
        target_tokens, _, _ = p.build_example(
            batch_size=c_slice.stop - c_slice.start,
            template=template,
            template_content=dict(query_name=vary_entity),
            context_content=target_context,
            prompt_id_start=c_slice.start,
            num_answers=num_entities*2,
        )
        _, source_cache = act_loc.get_normal_cache(
            model,
            input_tokens=source_tokens,
            prompt_ids=None, # not used
            stacked_token_maps=stacked_token_maps,
            answer_tokens=answer_tokens,
        )
        total_scores, _ = act_loc.discover_tokens(
            model=model,
            input_tokens=target_tokens,
            prompt_ids=slice(0, c_slice.stop - c_slice.start),
            stacked_token_maps=stacked_token_maps,
            answer_tokens=answer_tokens,
            answer=vary_entity,
            abnormal_run=partial(act_loc.switch_run, source_cache=source_cache),
        )
        total_scores_acc.append(total_scores * (c_slice.stop - c_slice.start))
    avg_total_score = sum(total_scores_acc) / num_samples
    context_start, context_end = stacked_token_maps['context_section'][0, :] # pick arbitary
    if target_layer is None:
        # dirty hack -- in theory we should try to skip localization when doing layer sweep anyway
        target_layer = 20
    peak_position = avg_total_score[context_start:context_end, target_layer-2: target_layer+3].sum(dim=1).argmax() + context_start
    return peak_position, avg_total_score

def get_class_probe(
    model, template, domain, localizations, num_entities, source_context, num_samples=100, batch_size=50, target_layer=20
):
    '''
    Returns:
        class_probe : FloatTensor[d_model]
    '''
    all_probes = []
    for c_slice in eu.generate_slices(0, num_samples, batch_size):
        source_tokens, answer_tokens, stacked_token_maps = p.build_example(
            batch_size=c_slice.stop - c_slice.start,
            template=template,
            template_content=dict(query_name=0),
            context_content=source_context,
            prompt_id_start=c_slice.start,
            num_answers=num_entities*2,
        )
        _, source_cache = act_loc.get_normal_cache(
            model,
            input_tokens=source_tokens,
            prompt_ids=None, # not used
            stacked_token_maps=stacked_token_maps,
            answer_tokens=answer_tokens,
        )
        averages = []
        for entity, pos in enumerate(localizations):
            if target_layer is None:
                averages.append(
                    torch.stack([
                        source_cache[f"blocks.{layer}.hook_resid_pre"][:, pos, :].mean(dim=0).cpu()
                        for layer in range(model.cfg.n_layers)
                    ])
                )
            else:
                averages.append(
                    source_cache[f"blocks.{target_layer}.hook_resid_pre"][:, pos, :].mean(dim=0)
                )

        all_probes.append(torch.stack(averages).mean(dim=0) * (c_slice.stop - c_slice.start))
    if target_layer is None:
        assert all_probes[0].shape == (model.cfg.n_layers , model.cfg.d_model)
    else:
        assert all_probes[0].shape == (model.cfg.d_model,)
    return sum(all_probes) / num_samples
        


def get_value_probe(
    model, template, domain, localizations, num_entities, source_context, num_samples=100, batch_size=50, target_layer=20
):
    all_value_probes = []
    for value in range(len(domain)):
        value_probes = []
        for vary_entity, pos in enumerate(localizations):
            for c_slice in eu.generate_slices(0, num_samples, batch_size):
                with condition_domain(domain, vary_entity, value):
                    source_tokens, answer_tokens, stacked_token_maps = p.build_example(
                        batch_size=c_slice.stop - c_slice.start,
                        template=template,
                        template_content=dict(query_name=0),
                        context_content=source_context,
                        prompt_id_start=c_slice.start,
                        num_answers=num_entities*2,
                    )
                _, source_cache = act_loc.get_normal_cache(
                    model,
                    input_tokens=source_tokens,
                    prompt_ids=None, # not used
                    stacked_token_maps=stacked_token_maps,
                    answer_tokens=answer_tokens,
                )
                if target_layer is None:
                    value_probes.append(
                        torch.stack([
                            source_cache[f"blocks.{layer}.hook_resid_pre"][:, pos, :].sum(dim=0).cpu()
                            for layer in range(model.cfg.n_layers)
                        ])
                    )
                else:
                    value_probes.append(
                        source_cache[f"blocks.{target_layer}.hook_resid_pre"][:, pos, :].sum(dim=0)
                    )
        value_probe = sum(value_probes) / num_samples / num_entities
        if target_layer is None:
            assert value_probe.shape == (model.cfg.n_layers , model.cfg.d_model)
        else:
            assert value_probe.shape == (model.cfg.d_model,)
        all_value_probes.append(value_probe)
    return all_value_probes

def save_localization(localized_pos, target_layer, total_scores, template, num_entities, vary_entity, source_context, save_dir):
    import seaborn as sns
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    input_tokens, token_map, _ = p.generate_tokenized_prompts(
        template=template,
        template_content=dict(query_name=0),
        context_content=source_context,
        prompt_id=0,
        num_answers=1
    )
    end_pos = token_map['context_section'].end
    labels = [str(i) + ': ' + template.tokenizer.decode(token) for i, token in enumerate(input_tokens)][:end_pos]
    fig = plt.figure(figsize=(15, 3))
    ax = fig.add_subplot()
    plot_scores = total_scores.cpu().float().numpy().T[:, :end_pos]
    sns.heatmap(
        plot_scores, ax=ax
    )
    ax.set_xticks(range(len(labels)), labels, rotation=32, rotation_mode='anchor',ha='right')
    ax.set_title(f"Localized position: {localized_pos} for entity {vary_entity} using layer {target_layer}")
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, f'localization_{vary_entity}_{type(template).__name__}.png')
        )
        # save raw scores
        np.save(
            os.path.join(save_dir, f'localization_{vary_entity}_{type(template).__name__}_scores.npy'),
            plot_scores
        )
        np.save(
            os.path.join(save_dir, f'localization_{vary_entity}_{type(template).__name__}_labels.npy'),
            labels
        )


def get_centroid_probe(
    model, template, domain, num_entities, source_context, target_context, target_layer, save_dir
):
    '''
    It's critical that domain is a property of template
    i.e. they're pointing to the same domains that template owns

    If target_layer is None, then compute probes for all layers
    '''
    # first, localize
    localizations = []
    for entity in range(num_entities):
        localized_pos, total_scores = localize_value(
            model=model,
            template=template,
            domain=domain,
            vary_entity=entity,
            num_entities=num_entities,
            source_context=source_context,
            target_context=target_context,
            target_layer=target_layer
        )
        save_localization(
            localized_pos=localized_pos, 
            target_layer=target_layer,
            total_scores=total_scores,
            template=template,
            num_entities=num_entities,
            vary_entity=entity,
            source_context=source_context,
            save_dir=save_dir
        )
        localizations.append(localized_pos)
    # get class centroids
    class_probe = get_class_probe(
        localizations=localizations,
        model=model,
        template=template,
        domain=domain,
        num_entities=num_entities,
        source_context=source_context,
        target_layer=target_layer
    )
    value_probes = get_value_probe(
        model=model,
        template=template,
        domain=domain,
        localizations=localizations,
        num_entities=num_entities,
        source_context=source_context,
        target_layer=target_layer
    )
    return class_probe, value_probes

def score_to_prediction(scores, threshold):
    best_scores, best_indices = scores.max(dim=0)
    best_indices[best_scores < threshold] = -1
    return best_indices


def results_to_values(mask, probe_results, probe):
    masked_probe_results = torch.where(mask.to(probe_results.device), probe_results, -1)
    all_values = []
    for batch in range(masked_probe_results.shape[0]):
        values = []
        for pos in range(masked_probe_results.shape[1]):
            if masked_probe_results[batch, pos] != -1:
                values.append((
                    probe.values[masked_probe_results[batch, pos].item()],
                    probe.type
                ))
        all_values.append(values)
    return all_values

class CentroidProbe(DomainProbe):
    def __init__(self, values, type):
        self.use_class_conditioned = True
        super().__init__(values, type)
    def sweep_layers(self, activations):
        '''
        Returns:
            scores: torch.FloatTensor[layer, len(self.values), batch_size, pos]
        
        '''
        value_probes = self.class_conditioned_value_probes if self.use_class_conditioned else self.value_probes
        all_scores = []
        for layer in range(activations.shape[1]):
            scores = torch.stack([
                einops.einsum(
                    activations[:, layer, :, :].cuda(),
                    probe[layer].cuda(),
                    'batch pos dim, dim -> batch pos',
                )
                for probe in value_probes
            ])
            all_scores.append(scores)
        return torch.stack(all_scores) # layer, values, batch, pos

    def __call__(self, activations, output_scores=False):
        '''
        Args:
            activations: torch.FloatTensor[batch_size, layer, pos, dim]
        Returns:
            classification: torch.LongTensor[batch_size, pos] in [-1, len(self.values)]
            scores: torch.FloatTensor[len(self.values), batch_size, pos]
        '''
        assert self.target_layer is not None
        value_probes = self.class_conditioned_value_probes if self.use_class_conditioned else self.value_probes
        scores = torch.stack([
            einops.einsum(
                activations[:, self.target_layer, :, :].cuda(),
                probe.cuda(),
                'batch pos dim, dim -> batch pos',
            )
            for probe in value_probes
        ])
        best_indices = score_to_prediction(scores, self.threshold)
        if output_scores:
            return best_indices, scores
        return best_indices
    
    def save(self, save_path):
        # logging.info(f'Saving {self.type} probe with values {self.values}')
        save_dict = {
            'value_probes': [probe.cpu() for probe in self.value_probes],
            'class_probe': self.class_probe.cpu(),
            'class_conditioned_value_probes': [probe.cpu() for probe in self.class_conditioned_value_probes],
            'threshold': self.threshold,
            'target_layer': self.target_layer,
        }
        torch.save(save_dict, save_path)
    @classmethod
    def load_or_none(cls, save_path):
        try:
            data = torch.load(save_path)
        except FileNotFoundError:
            return None
        if not all(key in data for key in ['value_probes', 'class_probe', 'class_conditioned_value_probes', 'threshold', 'target_layer']):
            return None
        probe = cls(target_layer=data['target_layer'], threshold=data['threshold'])
        # logging.info(f'Loading {probe.type} probe with values {probe.values}')
        probe.class_conditioned_value_probes = data['class_conditioned_value_probes']
        probe.value_probes = data['value_probes']
        probe.class_probe = data['class_probe']
        assert len(probe.class_conditioned_value_probes) == len(probe.values)
        return probe


class CountryCentroidProbe(CentroidProbe):
    def train(self, model, save_dir):
        num_entities = 2
        with tc.set_defaults(self.train_template, dict(
            prompt_type='chat_name_country',
            chat_style=self.chat_style
        )):
            class_probe, value_probes = get_centroid_probe(
                model=model,
                template=self.train_template,
                domain=self.train_template.capitals,
                num_entities=num_entities,
                source_context = [tc.Statement(i, i, None) for i in range(num_entities)],
                target_context = [tc.Statement(i, i + num_entities, None) for i in range(num_entities)],
                target_layer=self.target_layer,
                save_dir=save_dir
            )
        self.value_probes = [probe.cpu() for probe in value_probes]
        self.class_probe = class_probe.cpu()
        self.class_conditioned_value_probes = [
            probe.cpu() - class_probe.cpu()
            for probe in value_probes
        ]
    def __init__(self, *, target_layer=20, threshold=None, chat_style='llama_chat'):
        self.target_layer = target_layer
        self.threshold = threshold
        self.chat_style = chat_style
        self.train_template = ts.NameCountryTemplate('llama')
        super().__init__(self.train_template.capitals.data, self.train_template.capitals.type)

class NameCentroidProbe(CentroidProbe):
    def train(self, model, save_dir):
        num_entities = 2
        with tc.set_defaults(self.train_template, dict(
            prompt_type='chat_country_name',
            chat_style=self.chat_style
        )):
            class_probe, value_probes = get_centroid_probe(
                model=model,
                template=self.train_template,
                domain=self.train_template.names,
                num_entities=num_entities,
                source_context = [tc.Statement(i, i, None) for i in range(num_entities)],
                target_context = [tc.Statement(i + num_entities, i, None) for i in range(num_entities)],
                target_layer=self.target_layer,
                save_dir=save_dir
            )
        self.value_probes = [probe.cpu() for probe in value_probes]
        self.class_probe = class_probe.cpu()
        self.class_conditioned_value_probes = [
            probe.cpu() - class_probe.cpu()
            for probe in value_probes
        ]
    def __init__(self, *, target_layer=20, threshold=None, chat_style='llama_chat'):
        self.target_layer = target_layer
        self.threshold = threshold
        self.chat_style = chat_style
        self.train_template = ts.NameCountryTemplate('llama')
        super().__init__(self.train_template.names.data, self.train_template.names.type)

class FoodCentroidProbe(CentroidProbe):
    def train(self, model, save_dir):
        num_entities = 2
        with tc.set_defaults(self.train_template, dict(
            prompt_type='chat_name_food',
            chat_style=self.chat_style
        )):
            class_probe, value_probes = get_centroid_probe(
                model=model,
                template=self.train_template,
                domain=self.train_template.foods,
                num_entities=num_entities,
                source_context = [tc.Statement(i, i, None) for i in range(num_entities)],
                target_context = [tc.Statement(i, i + num_entities, None) for i in range(num_entities)],
                target_layer=self.target_layer,
                save_dir=save_dir
            )
        self.value_probes = [probe.cpu() for probe in value_probes]
        self.class_probe = class_probe.cpu()
        self.class_conditioned_value_probes = [
            probe.cpu() - class_probe.cpu()
            for probe in value_probes
        ]
    def __init__(self, *, target_layer=20, threshold=None, chat_style='llama_chat'):
        self.target_layer = target_layer
        self.threshold = threshold
        self.chat_style = chat_style
        self.train_template = ts.NameFoodTemplate('llama')
        super().__init__(self.train_template.foods.data, self.train_template.foods.type)


class OccupationCentroidProbe(CentroidProbe):
    def train(self, model, save_dir):
        num_entities = 2
        with tc.set_defaults(self.train_template, dict(
            prompt_type='chat_name_occupation',
            chat_style=self.chat_style
        )):
            class_probe, value_probes = get_centroid_probe(
                model=model,
                template=self.train_template,
                domain=self.train_template.occupations,
                num_entities=num_entities,
                source_context = [tc.Statement(i, i, None) for i in range(num_entities)],
                target_context = [tc.Statement(i, i + num_entities, None) for i in range(num_entities)],
                target_layer=self.target_layer,
                save_dir=save_dir
            )
        self.value_probes = [probe.cpu() for probe in value_probes]
        self.class_probe = class_probe.cpu()
        self.class_conditioned_value_probes = [
            probe.cpu() - class_probe.cpu()
            for probe in value_probes
        ]
    def __init__(self, *, target_layer=20, threshold=None, chat_style='llama_chat'):
        self.target_layer = target_layer
        self.threshold = threshold
        self.chat_style = chat_style
        self.train_template = ts.NameOccupationTemplate('llama')
        super().__init__(self.train_template.occupations.data, self.train_template.occupations.type)

