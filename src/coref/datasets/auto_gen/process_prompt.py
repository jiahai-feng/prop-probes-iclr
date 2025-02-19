from openai import OpenAI
import argparse
import coref
import os
import yaml
import openai
from coref import COREF_ROOT
from pathlib import Path

# imports
import random
import time
from coref.datasets.auto_gen.gpt import get_completions, scrub_quotes

import coref.run_manager as rm
import json

from dataclasses import dataclass
from typing import Literal, Any, Optional
@dataclass(kw_only=True)
class Cfg(rm.Cfg):
    prompt_path: str
    model_name: str = 'gpt-3.5-turbo'
    gpt_instruct_path: str
    prompt_header: str = 'prompt'


f'''
python -m coref.datasets.auto_gen.process_prompt \
    --prompt_path exports/datasets/name_country_food_fixed/dataset.json \
    --output_dir exports/datasets/name_country_food_fixed/es_translation/ \
    --gpt_instruct_path coref/datasets/auto_gen/rewrite_prompts/translate_to_es.txt


python -m coref.datasets.auto_gen.process_prompt \
    --prompt_path exports/datasets/name_country_food_basic/dataset.json \
    --output_dir exports/datasets/name_country_food_basic/indirect/ \
    --gpt_instruct_path coref/datasets/auto_gen/rewrite_prompts/indirect_paraphrase.txt


python -m coref.datasets.auto_gen.process_prompt \
    --prompt_path exports/datasets/name_country_food_basic/dataset.json \
    --output_dir exports/datasets/name_country_food_basic/paraphrase/ \
    --gpt_instruct_path coref/datasets/auto_gen/rewrite_prompts/paraphrase.txt
'''

@rm.automain
def main(cfg, output_dir):
    cfg = Cfg(**cfg)
    with open(Path(COREF_ROOT) / cfg.prompt_path, 'r') as f:
        all_prompts = json.load(f)
    with open(Path(COREF_ROOT) / cfg.gpt_instruct_path, 'r') as f:
        instruct_prompt = f.read()
    client = OpenAI()

    sep = "<|sep|>"
    for prompt in all_prompts:
        if cfg.prompt_header == 'prompt':
            context, _ = prompt['prompt'].split(sep)
        else:
            context = prompt[cfg.prompt_header]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruct_prompt.format(input=context)},
        ]
        response = get_completions(
            client=client,
            model=cfg.model_name,
            messages=messages
        )
        prompt['context'] = scrub_quotes(response.choices[0].message.content)
    with open(output_dir / "dataset.json", 'w') as f:
        json.dump(all_prompts, f, indent=2)
