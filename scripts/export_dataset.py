import coref.datasets.api as ta
from functools import partial
import numpy as np
import torch
import einops
import logging

from dataclasses import dataclass
from typing import Literal, Any, Optional
import coref.run_manager as rm
import json

@dataclass(kw_only=True)
class Cfg(rm.Cfg):    
    template: str = "NameCountryTemplate"
    prompt_type: Optional[str] = None
    chat_style: Optional[str] = None
    context_type: Any = None
    num_entities: int
    num_samples: int
    prompt_id_start: int = 0
    split_sep: bool = False

@rm.automain
def main(cfg, output_dir):
    cfg = Cfg(**cfg)
    if cfg.context_type is not None:
        try:
            cfg.context_type = int(cfg.context_type)
        except ValueError:
            pass
    template = ta.get_template(cfg.template)('llama')
    stuff = []
    for prompt_id in range(cfg.prompt_id_start, cfg.prompt_id_start + cfg.num_samples):
        content_context = template.get_standard_context(cfg.num_entities)
        query_name = prompt_id % cfg.num_entities
        prompt = ta.generate_prompts(
            template=template,
            template_content=dict(
                query_name=query_name,
                prompt_type=cfg.prompt_type,
                chat_style=cfg.chat_style,
                context_type=cfg.context_type,
            ),
            context_content=content_context,
            prompt_id=prompt_id,
            num_answers=cfg.num_entities,
        )
        ctx = {} if not cfg.split_sep else {
            'context': prompt['prompt'].split('<|sep|>')[0].strip()
        }
        stuff.append({
            "prompt": prompt['prompt'],
            "prompt_id": prompt_id,
            "answers": [template.tokenizer.decode(x) for x in prompt['answers']],
            "query_name": query_name,
            "predicates": template.get_predicates(content_context, prompt_id),
            **ctx
        })
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(stuff, f, indent=2)

        

    
    