from distutils.util import strtobool
from importlib import reload
import os
from coref import COREF_ROOT
from coref.utils import cache
from functools import partial
import itertools
import coref.run_manager as rm
import torch
import coref.hessians as ch
import coref.models as models
from functools import partial
import numpy as np
import torch
import einops
import logging

import coref.datasets.templates.simple as ts
import coref.datasets.templates.common as tc
import coref.datasets.templates.triplet as tt
import coref.datasets.api as ta
import coref.parameters as p
import coref.datascience as ds
import coref.expt_utils as eu

from dataclasses import dataclass
from typing import Literal, Any, Optional

@dataclass(kw_only=True)
class Cfg(rm.Cfg):
    model: str
    num_devices: int = 1
    is_hf: bool = False

    hessian_mode: Literal["finite", "point"]
    name_width: int = 1
    attr_width: int = 1
    
    template: str = "NameCountryTemplate"
    prompt_type: Optional[str] = None
    prompt_id_start: int = 0

    # finite hessian
    n_samples: int = 4096
    num_prompts: int = 20
    name_rms: float = 1e-3
    attr_rms: float = 1e-3

    # point hessian
    swap_dir: bool = False

    # ablation flags
    uniform_scale: bool = False
    interpolating_factor: float = 0.5





@rm.automain
def main(args, output_dir):
    args = Cfg(**args)
    args.save(output_dir / "args.yaml")

    train_template = ta.get_template(args.template)('llama', 'train')
    test_template = ta.get_template(args.template)('llama', 'test')
    assert args.hessian_mode in ['finite', 'point']

    model = models.fetch_model(args.model, num_devices=args.num_devices, dtype=torch.bfloat16, hf=args.is_hf)
    # used to be float16

    
    ctxt = ch.prepare_stuff(
        model=model,
        train_template=train_template,
        prompt_type=args.prompt_type,
        interpolating_factor=args.interpolating_factor
    )
    func = lambda x, y: ch.evaluate_f(
        namespace=ctxt,
        test_template=test_template,
        prompt_id_start=args.prompt_id_start,
        num_samples=args.num_prompts,
        uniform_scale=args.uniform_scale,
        x=x,
        y=y
    )
    if args.hessian_mode == 'finite':
        output = ch.finite_hessian(
            func, model, n_samples=args.n_samples, widths=(args.name_width, args.attr_width), rms_vals=(args.name_rms, args.attr_rms))
        torch.save(output, output_dir / 'hessian.pt')
    elif args.hessian_mode == 'point':
        output = ch.point_hessian(
            func, model, widths=(args.name_width, args.attr_width), swap_dir=args.swap_dir
        )
        torch.save(output, output_dir / 'hessian.pt')
    else:
        raise Exception(f'Unknown hessian mode {args.hessian_mode}')
    logging.info('done')
    


# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser()    
#     parser.add_argument(
#         "--config",
#         type=str,
#         required=True,
#     )
#     parser.add_argument(
#         "--output_root",
#         type=str,
#         default=os.path.join(COREF_ROOT, 'runs')
#     )
#     parser.add_argument(
#         "--experiments_root",
#         type=str,
#         default=os.path.join(COREF_ROOT, 'experiments')
#     )
#     args = parser.parse_args()
#     cfg, output_dir = rm.load_and_setup(args.config, args.output_root, args.experiments_root)

#     main(cfg, output_dir)


    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     help=f'choices: {", ".join(models.LLAMA_2_MODELS)}',
    #     required=True,
    # )
    # parser.add_argument("--num_devices", type=int, default=1)
    # parser.add_argument("--is_hf", action="store_true")
    # parser.add_argument("--hessian_mode", type=str)
    # # func params
    # parser.add_argument("--name_width", type=int, default=1)
    # parser.add_argument("--attr_width", type=int, default=1) 
    # parser.add_argument("--prompt_type", type=str, default=None)
    # parser.add_argument("--template", type=str, default='NameCountryTemplate')
    # parser.add_argument("--prompt_id_start", type=int, default=0)
    # # finite hessian params
    # parser.add_argument("--n_samples", type=int, default=4096)
    # parser.add_argument("--num_prompts", type=int, default=1)
    # parser.add_argument("--name_rms", type=float, default=2e-3) # x
    # parser.add_argument("--attr_rms", type=float, default=2e-3) # y
    # # point hessian params
    # parser.add_argument("--swap_dir", type=lambda x: bool(strtobool(x)), default=False)

    # # options
    # parser.add_argument("--cache_path", default=os.path.join(COREF_ROOT, "cache"))
    # parser.add_argument("--force_compute", action="store_false")
    # parser.add_argument("--force_time", nargs="?", type=str)
    # parser.add_argument("--dry_run", action="store_true")

