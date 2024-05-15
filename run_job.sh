#!/bin/bash 
#SBATCH --job-name=llama_run
#SBATCH --time=0-30:00
#SBATCH --partition=gpu-4-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

# informando ao tch-rs que desejo compilar com cuda na versão 11.7
export TORCH_CUDA_VERSION=cu117

torchrun llama2/hub-inf.py llama2/llama-2-7b/ llama2/tokenizer.model
