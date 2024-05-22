#!/bin/bash 
#SBATCH --job-name=llama_run
#SBATCH --time=0-30:00
#SBATCH --partition=gpu-8-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5

# informando ao tch-rs que desejo compilar com cuda na versão 11.7
export TORCH_CUDA_VERSION=cu117

torchrun llama3/inference.py \
	--ckpt_dir llama3/Meta-Llama-3-8B/ \
	--tokenizer_path llama3/Meta-Llama-3-8B/tokenizer.model
