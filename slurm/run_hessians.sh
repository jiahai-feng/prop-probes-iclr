#!/bin/bash
#SBATCH -c 16
#SBATCH --output=jobs/hessians-%j.out
python  -m scripts.run_hessians \
    --config $CONFIG_FILE
