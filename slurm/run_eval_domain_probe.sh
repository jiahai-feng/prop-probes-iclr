#!/bin/bash
#SBATCH --output=jobs/eval_domain_probe-%j.out
#SBATCH -c 16
#SBATCH --gres=gpu:8

cd /accounts/projects/jsteinhardt/fjiahai/coref

/accounts/projects/jsteinhardt/fjiahai/.conda/envs/coref/bin/python -m scripts.run_eval_domain_probe \
    --config $CONFIG_FILE