#!/bin/bash
#SBATCH --output=jobs/vsb-%j.out

cd /accounts/projects/jsteinhardt/fjiahai/coref

python  -m coref.vector_subspace_baseline \
    --config $CONFIG_FILE
