#!/bin/bash
#SBATCH --output=jobs/eval_form-%j.out
python  -m scripts.run_eval_form \
    --config $CONFIG_FILE
