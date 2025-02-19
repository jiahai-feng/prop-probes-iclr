#!/bin/bash
#SBATCH --output=jobs/genderbias-%j.out
#SBATCH -c 16

python -m coref.gender.synthetic \
    --config $CONFIG_FILE