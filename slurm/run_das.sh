#!/bin/bash
#SBATCH --output=jobs/train_das-%j.out
#SBATCH -c 16
#SBATCH --gres=gpu:4


python -m coref.train_das \
    --config $CONFIG_FILE