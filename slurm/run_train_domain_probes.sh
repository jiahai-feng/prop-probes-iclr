#!/bin/bash
#SBATCH --output=jobs/train_domain_probe-%j.out
#SBATCH -c 16
#SBATCH --gres=gpu:4

python -m scripts.train_domain_probes \
    --config $CONFIG_FILE