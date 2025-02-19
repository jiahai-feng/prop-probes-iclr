#!/bin/bash
#SBATCH --output=jobs/eval_das-%j.out
#SBATCH -c 16
#SBATCH --gres=gpu:2

python -m scripts.eval_das \
    --config $CONFIG_FILE