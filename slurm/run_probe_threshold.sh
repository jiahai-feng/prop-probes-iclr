#!/bin/bash
#SBATCH --output=jobs/threshold_probe-%j.out
#SBATCH -c 16

python -m scripts.probe_threshold \
    --expts_root $EXPTS_ROOT \
    --outputs_root $OUTPUTS_ROOT