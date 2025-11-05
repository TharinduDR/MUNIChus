#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --gres=gpu:nvidia_rtx_a5000:3
#SBATCH --mem=80G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home
huggingface-cli login --token

python -m mllm.qwen3



