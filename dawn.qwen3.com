#!/bin/bash
#SBATCH --job-name=headline_generation-gpu
#SBATCH --account=AIRR-P39-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:3 # Number of requested GPUs per node
#SBATCH --time=12:00:00              # total run time limit (HH:MM:SS)

module purge
module load rhel8/default-dawn
module load intelpython-conda/2025.0

conda activate /home/dn-rana1/rds/conda_envs/llm_exp

export HF_HOME=//home/dn-rana1/rds/rds-airr-p39-JpwWyPZa2Oc/hf_home/
export HF_TOKEN=""

python -m mllm.qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct



