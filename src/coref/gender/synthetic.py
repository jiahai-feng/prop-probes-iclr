import os

import json
from functools import partial

import torch
import pandas as pd
import einops



import  coref.models as models

from coref import COREF_ROOT
import coref.datasets.templates.common as tc
import coref.datasets.templates.simple as ts
import coref.datasets.api as ta
import coref.parameters as p

import coref.expt_utils as eu
import coref.interventions as cint
import coref.probes.lookup_probe as lkp


def get_template_and_contexts(chat_style):
    pro_contexts = [
        [
            tc.Statement(0, 0, 'occupation'),
            tc.Statement(1, 1, 'occupation'),
            tc.Statement(0, 0, 'capital'), # male
            tc.Statement(1, 1, 'capital'), # female
        ],
        [
            tc.Statement(0, 0, 'occupation'),
            tc.Statement(1, 1, 'occupation'),
            tc.Statement(1, 1, 'capital'), # female
            tc.Statement(0, 0, 'capital'), # male
        ],
    ]
    anti_contexts = [
        [
            tc.Statement(0, 0, 'occupation'),
            tc.Statement(1, 1, 'occupation'),
            tc.Statement(1, 0, 'capital'), # female
            tc.Statement(0, 1, 'capital'), # male
        ],
        [
            tc.Statement(0, 0, 'occupation'),
            tc.Statement(1, 1, 'occupation'),
            tc.Statement(0, 1, 'capital'), # male
            tc.Statement(1, 0, 'capital'), # female
        ],
    ]

    # provide answers only for internal eval
    # external eval is always [0, 1] for pro, and [1, 0] for anti

    pro_answers = [ # the i-th occupation should be matched with j-th gender
        [0, 1], # cis
        [1, 0], # trans
    ]
    anti_answers = [
        [0, 1], # cis
        [1, 0], # trans
    ]
    template = ts.GenderCapitalOccupationTemplate('llama')
    template.default_template_content['chat_style'] = chat_style
    return dict(
        pro_answers=pro_answers,
        anti_answers=anti_answers,
        pro_contexts=pro_contexts,
        anti_contexts=anti_contexts,
        template=template
    )
def eval_external(model, pro_contexts, anti_contexts, pro_answers, anti_answers, template):
    pro_results = eu.repeat(2, lambda query_gender:
                    torch.cat([
        p.evaluate_interventions(
            model=model,
            num_samples=200,
            template=template,
            context_content=context,
            template_content=dict(
                query_name=query_gender,
                chat_style='tulu_chat'
            ),
            num_answers=2,
            prompt_id_start=0,
            short_circuit=True,
            return_raw=True
        )
            for context in pro_contexts
        ])
    )
    anti_results = eu.repeat(2, lambda query_gender:
                    torch.cat([
        p.evaluate_interventions(
            model=model,
            num_samples=200,
            template=template,
            context_content=context,
            template_content=dict(
                query_name=query_gender,
                chat_style='tulu_chat'
            ),
            num_answers=2,
            prompt_id_start=0,
            short_circuit=True,
            return_raw=True
        )
            for context in anti_contexts
        ])
    )


    return {
        'pro_results': pro_results, # [query occupation, batch, gender]
        'anti_results': anti_results,
        'pro_answers': torch.tensor([0,1])[:, None].expand(-1, pro_results.shape[1]),
        'anti_answers': torch.tensor([1,0])[:, None].expand(-1, anti_results.shape[1]),
    }

def get_accuracy(results, answers):
    '''
    Args
        results : [query occupation, batch, gender]
        answers : [query occupation, batch]
    Returns:
        accuracy: float - the fraction of correct predictions
        calibrated_accuracy: float - the fraction of the time that \
            when forced to choose between [0, 1] and [1, 0], the model \
            picks correctly
    '''
    cis_minus_trans = results[0, :, 0] + results[1, :, 1] - results[0, :, 1] - results[1, :, 0]
    is_cis = answers[0] == 0
    calib_acc = torch.where(
        is_cis,
        cis_minus_trans > 0,
        cis_minus_trans < 0
    )
    return {
        'accuracy': results.argmax(dim=-1).eq(answers).float().mean().item(),
        'calibrated_accuracy': calib_acc.float().mean().item()
    }
    


def cache_to_norm_acts(model, cache):
    acts = einops.rearrange(
        torch.stack([cache[f'blocks.{layer}.hook_resid_pre'].cpu() for layer in range(model.cfg.n_layers)]),
        'layer batch pos dim -> batch layer pos dim'
    )
    norm_acts = acts / acts.norm(dim=-1, keepdim=True)
    return norm_acts

def get_overlap_scores(input_tokens, answer_tokens, stacked_token_maps, prompt_ids, model, affinity_fn):
    logits, cache = cint.run_cached_model( 
        execution_fn=lambda: model.get_logits(input_tokens),
        model=model,
        names_filter=lambda x: "hook_resid_pre" in x,
        incl_bwd=False,
    )
    occupation_locations = [
        stacked_token_maps['context'][i]['name'][0][:, 1] - 1
        for i in [0, 1]
    ]
    gender_locations = [
        stacked_token_maps['context'][i]['name'][0][:, 1] - 1
        for i in [2, 3]
    ]
    norm_acts = cache_to_norm_acts(model, cache)
    scores = affinity_fn(norm_acts .float()) # batch, pos1, pos2
    rets = [[None, None], [None, None]]
    for io, occ_locs in enumerate(occupation_locations):
        for ig, gen_locs in enumerate(gender_locations):
            assert occ_locs.shape == (input_tokens.shape[0],)
            assert gen_locs.shape == (input_tokens.shape[0],)
            rets[io][ig] = scores[
                torch.arange(occ_locs.shape[0]),
                occ_locs,
                gen_locs
            ]
    return einops.rearrange(
        torch.stack([torch.stack(x) for x in rets]),
        'occupation gender batch -> occupation batch gender'
    ) # matches shape of external eval
            

def internal_scores(context, template, answer, model, affinity_fn, num_samples):
    overlap_scores = p.run_over_batches(
        model=model,
        num_samples=num_samples,
        batch_size=50,
        template=template,
        context_content=context,
        template_content=dict(
            query_name=0,
            chat_style='tulu_chat'
        ),
        num_answers=2,
        prompt_id_start=0,
        custom_runner=partial(get_overlap_scores, model=model, affinity_fn=affinity_fn)
    )

    overlap_scores = torch.cat(overlap_scores, dim=1)

    return overlap_scores

def eval_internal(model, template, pro_contexts, anti_contexts, pro_answers, anti_answers, affinity_fn):
    num_samples = 200
    pro_results = torch.cat([
        internal_scores(
            context=context,
            template=template,
            answer=answer,
            model=model,
            affinity_fn=affinity_fn,
            num_samples=num_samples
        )
        for context, answer in zip(pro_contexts, pro_answers)
    ], dim=1)
    anti_results = torch.cat([
        internal_scores(
            context=context,
            template=template,
            answer=answer,
            model=model,
            affinity_fn=affinity_fn,
            num_samples=num_samples
        )
        for context, answer in zip(anti_contexts, anti_answers)
    ], dim=1)
    return {
        'pro_results': pro_results.float().cpu(),
        'anti_results': anti_results.float().cpu(),
        'pro_answers' : torch.cat([
            torch.tensor(answers)[:, None].expand(-1, num_samples)
            for answers in pro_answers
        ], dim=1),
        'anti_answers' : torch.cat([
            torch.tensor(answers)[:, None].expand(-1, num_samples)
            for answers in anti_answers
        ], dim=1)
    }

def compute_metrics(data):
    return [
        {
            'split': 'pro',
            **get_accuracy(data['pro_results'], data['pro_answers'])
        },
        {
            'split': 'anti',
            **get_accuracy(data['anti_results'], data['anti_answers'])
        }
    ]




import coref.run_manager as rm
import json

import logging
from dataclasses import dataclass, field
from typing import Literal, Any, Optional, Dict
@dataclass(kw_only=True)
class Cfg(rm.Cfg):
    model: str
    num_devices: int = 1
    is_hf: bool = True
    peft_dir: Optional[str] = None

    chat_style: Literal['llama_chat', 'tulu_chat'] = 'llama_chat'

    form_path: str
    form_type: str


    affinity_fn: str = "U_subspace_sq_affinity_fn"
    affinity_fn_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            'dim': 50,
            'layer': 15
        }
    )



@rm.automain
def main(cfg, output_dir):
    logging.output_dir = output_dir
    cfg = Cfg(**cfg)
    model = models.fetch_model(cfg.model, num_devices=cfg.num_devices, dtype=torch.bfloat16, hf=cfg.is_hf, peft_dir=cfg.peft_dir)
    
    affinity_fn = lkp.get_affinity_fn(
        affinity_fn=cfg.affinity_fn,
        form_path=cfg.form_path,
        form_type=cfg.form_type,
        affinity_fn_kwargs=cfg.affinity_fn_kwargs
    )

    t_and_cs = get_template_and_contexts('tulu_chat')
    external_data = eval_external(
        model=model,
        **t_and_cs
    )
    external_metrics = compute_metrics(external_data)

    internal_data = eval_internal(
        model=model,
        affinity_fn=affinity_fn,
        **t_and_cs
    )

    internal_metrics = compute_metrics(internal_data)

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump([
            {
                **met,
                'method': 'external'
            }
            for met in external_metrics
        ] + [
            {
                **met,
                'method': 'internal'
            }
            for met in internal_metrics
        ], f)
    torch.save({
        'external_data': external_data,
        'internal_data': internal_data
    }, output_dir / 'data.pt')