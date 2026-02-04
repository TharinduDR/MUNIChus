#!/bin/bash
#SBATCH --output=output.txt
#SBATCH --error=error.txt

#SBATCH --partition=cpu-48h
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=h.hettiarachchi@lancaster.ac.uk

##export HF_HOME=
##huggingface-cli login --token

python -m xl_stats



