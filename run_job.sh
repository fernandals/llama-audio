#!/bin/bash 
#SBATCH --job-name=llama2_test
#SBATCH --time=1-23:58
#SBATCH --partition=gpu-4-a100
#SBATCH --gres=gpu:1

# informando ao tch-rs que desejo compilar com cuda na versão 11.7
export TORCH_CUDA_VERSION=cu117

torchrun example_text_completion.py llama-2-7b/ tokenizer.model
