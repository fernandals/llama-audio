#!/bin/bash 
#SBATCH --job-name=llama2_test
#SBATCH --time=1-23:58
#SBATCH --partition=gpu-4-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

# informando ao tch-rs que desejo compilar com cuda na versão 11.7
export TORCH_CUDA_VERSION=cu117

torchrun fine_tune_audio.py llama-2-7b/ tokenizer.model
