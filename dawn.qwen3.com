#!/bin/bash
#SBATCH --job-name=headline_generation-gpu
#SBATCH --account=AIRR-P39-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:3 # Number of requested GPUs per node
#SBATCH --time=12:00:00              # total run time limit (HH:MM:SS)


ulimit -n 65536

module purge
module load rhel8/default-dawn
module load intelpython-conda/2025.0

conda activate /home/dn-rana1/rds/conda_envs/llm_exp

# Intel XPU stability environment variables
export IPEX_TILE_AS_DEVICE=0
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export NEOReadDebugKeys=1
export ClDeviceGlobalMemSizeAvailablePercent=95

export HF_HOME=//home/dn-rana1/rds/rds-airr-p39-JpwWyPZa2Oc/hf_home/
export HF_TOKEN=""

python -m mllm.dawn_qwen3 --model_id Qwen/Qwen3-VL-8B-Instruct



