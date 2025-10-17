#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home
huggingface-cli login --token

python -m train.aya_qlora_finetune \
    --output_dir ./aya-vision-qlora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --eval_steps 100 \
    --save_steps 100