#!/bin/bash
#SBATCH --output=jobs/eval_probe-%j.out
#SBATCH -c 16
#SBATCH --gres=gpu:8

python -m coref.probes.evaluate \
    --config $CONFIG_FILE