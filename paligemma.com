#!/bin/bash
#SBATCH --partition=a5000-6h
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home
huggingface-cli login --token

python -m baseline.paligemma


