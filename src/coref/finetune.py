import coref.datasets.templates.common as tc
from tqdm import trange

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

from exults.log_utils import Logger

from coref import COREF_ROOT
import coref.models as models

def tokenize_and_mask_batch(batch, tokenizer):
    '''
    Args:
        batch: List[Tuple[text: str, target: str]]
    Returns:
        inputs: Dict[str, torch.Tensor]
            - Contains at least input_ids, attention_mask, labels
            - Feed into model forward pass
            - labels are masked out to only include the target
    '''
    strings = []
    refs = []
    for text, target in batch:
        string, ref = tc.TrackFormatter().format('{text} {target}', text=text, target=target)
        strings.append(string)
        refs.append(ref)
    inputs = tokenizer(
        strings,
        return_tensors='pt',
        padding=True,
        return_offsets_mapping=True
    )
    labels = inputs['input_ids'].clone()
    for i, ref in enumerate(refs):
        token_ref = tc.recursive_align_tokens(ref, inputs['offset_mapping'][i, inputs['attention_mask'][i].bool()])
        token_positions = torch.arange(inputs['attention_mask'].shape[1])[inputs['attention_mask'][i].bool()]
        target_start, target_end = token_ref['target'][0]
        target_start = token_positions[target_start] if target_start < len(token_positions) else token_positions[-1]+1
        target_end = token_positions[target_end] if target_end < len(token_positions) else token_positions[-1]+1
        labels[i, : target_start] = -100
        labels[i, target_end :] = -100
        if tokenizer.decode(labels[i, target_start : target_end]).strip() != batch[i][1].strip():
            print('Warning: target mismatch')
            print(tokenizer.decode(labels[i, target_start : target_end]))
            print(batch[i][1])
    inputs['labels'] = labels
    del inputs['offset_mapping']
    return inputs



def prepare_model_for_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # "lm_head",
        ],
        bias="none",
        lora_dropout=0.1,  # Conventional
        task_type="CAUSAL_LM",
    )


    # model.gradient_checkpointing_disable()
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_disable()

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    print_trainable_parameters(model)
    
    return model

def prepare_model_for_finetune(model):
    frozen_modules = [model.model.embed_tokens, model.lm_head]
    for mod in frozen_modules:
        for param in mod.parameters():
            param.requires_grad = False
    
    return model

from math import cos, pi
from functools import partial
def linear_scheduler(step, max_steps, initial_factor, final_factor):
    alpha = step / max_steps
    return alpha * final_factor + (1-alpha) * initial_factor
    
def cosine_scheduler(step, warmup_steps, max_steps, final_factor, initial_factor):
    if step < warmup_steps:
        return linear_scheduler(step, warmup_steps, initial_factor, 1.)
    elif step >= max_steps:
        return final_factor
    else:
        step = step - warmup_steps
        max_steps = max_steps - warmup_steps
        return final_factor + (1. - final_factor) * (1 + cos(pi * step / max_steps)) / 2

@torch.enable_grad()
def finetune(model, tokenizer, ds, logger):
    batch_size = 32
    num_grad_acc_steps = 4
    num_epochs = 1
    num_total_steps = num_epochs * len(ds) // (batch_size * num_grad_acc_steps)
    warmup_steps = 10
    lr = 1e-4
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        betas=(0.9, 0.95),
        lr=lr
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optim, partial(
        cosine_scheduler, 
        warmup_steps=warmup_steps,
        max_steps=num_total_steps,
        initial_factor=1e-4,
        final_factor=0.
    ))
    # sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: 1.)
    logger.reset()
    logger.init('lrs')
    logger.init('train_losses')
    grad_update_counter = num_grad_acc_steps
    optim.zero_grad()
    for epoch in range(num_epochs):
        train_perm = torch.randperm(len(ds))
        with trange(0, len(ds), batch_size) as pbar:
            for start in pbar:
                # print(sched.get_last_lr())
                end = min(len(ds), start + batch_size)
                inputs = tokenize_and_mask_batch([ds[i] for i in train_perm[start: end]], tokenizer)
                outputs = model(**inputs)
                outputs.loss.backward()
                grad_update_counter -= 1
                if grad_update_counter == 0:
                    grad_update_counter = num_grad_acc_steps
                    optim.step()
                    sched.step()
                    optim.zero_grad()
                    
                    logger.train_losses.append(outputs.loss.item())
                    pbar.set_description(f'Loss: {logger.train_losses[-1]:.2f}, {sched.get_last_lr()[0]:.3e}')
                    logger.lrs.append(sched.get_last_lr()[0])
                
            
        print(f'Last train_loss: {logger.train_losses[-1]}')

import coref.run_manager as rm
import json

from dataclasses import dataclass, field
from typing import Literal, Any, Optional, Dict
@dataclass(kw_only=True)
class Cfg(rm.Cfg):
    model: str
    train_mode: Literal['lora', 'finetune'] = 'finetune'
    save_dir: str

    finetune_ds: str = 'exports/datasets/finetune/fixed_es_wrong.json'


@rm.automain
def main(cfg, output_dir):
    logging.output_dir = output_dir
    cfg = Cfg(**cfg)

    model = models.fetch_model(cfg.model, dtype=torch.bfloat16, hf=True)
    with open(os.path.join(COREF_ROOT, cfg.finetune_ds), 'r') as f:
        finetune_ds = json.load(f)
    logger = Logger(output_dir / 'logs')
    if cfg.train_mode == 'lora':
        model = prepare_model_for_lora(model)
    else:
        model = prepare_model_for_finetune(model)

    model.tokenizer.pad_token = model.tokenizer.bos_token

    finetune(model, model.tokenizer, finetune_ds, logger)

    model.save_pretrained(cfg.save_dir)