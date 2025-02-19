from functools import partial
import logging
import os
import numpy as np
import torch
import einops
import torch.nn.functional as F
from pathlib import Path
import gc
import json

from coref import COREF_ROOT

import coref.probes.lookup_probe as lkp

from coref.datasets.templates.basic import BasicPrefixTemplate
import coref.datasets.api as ta

def export_finetuning_ds(dataset_path, output_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    for d in dataset:
        if not all(key in d for key in ['context', 'prompt_id', 'predicates']):
            raise ValueError('Dataset must have keys context, prompt_id, predicates')
        if 'prefix' not in d:
            d['prefix'] = ''
        d['predicates'] = lkp.list_to_tuple(d['predicates'])
    template = BasicPrefixTemplate('llama')
    def get_other(preds, attr, dtype):
        r = [o_attr for _, o_attr, o_dtype in preds
            if dtype == o_dtype and o_attr != attr
            ]
        return r[0]
    new_dataset = []
    for d in dataset:
        for pred in d['predicates']:
            name, attr, dtype = pred
            if dtype == 'FOODS':
                question, response = lkp.PromptDecoder.stage3['food']
                format_attr = lambda x: x
            elif dtype == 'CAPITALS':
                question, response = lkp.PromptDecoder.stage3['country']
                format_attr = lambda x: x[0]
            elif dtype == 'OCCUPATIONS':
                question, response = lkp.PromptDecoder.stage3['occupation']
                format_attr = lambda x: x
            else:
                assert False, f'Invalide predicate {pred}'

            question = question.format(name=name)
            response = response.format(name=name)
            prompts = ta.generate_prompts(
                template=template,
                template_content=dict(
                    prefix='',
                    context=d['context'] + question,
                    query=response
                ),
                context_content=None,
                prompt_id=0,
                num_answers=0
            )
            new_dataset.append({
                'prompt': prompts['prompt'],
                'correct_answer': format_attr(attr),
                'wrong_answer': format_attr(get_other(d['predicates'], attr, dtype))
            })
                
    finetune_ds = [
        (
            d['prompt'],
            d['wrong_answer']
        )
        for d in new_dataset
    ]
    with open(output_path, 'w') as f:
        json.dump(finetune_ds, f)

if __name__ == '__main__':
    dataset_path = os.path.join(COREF_ROOT, "exports/datasets/name_country_food_occupation_basic_val/es_translation/dataset.json")
    output_path = os.path.join(COREF_ROOT, 'exports/datasets/finetune/fixed_es_wrong.json')
    export_finetuning_ds(dataset_path, output_path)