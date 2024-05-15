# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# LLaMA 2 inference connected with HuBERT audio encoder (using HF dataset)

import fire

from llama import Llama
from typing import List

import torch, torchaudio
from torch import nn

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1100,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    from datasets import load_dataset
    fleurs = load_dataset("google/fleurs", "en_us", split="test")
    audio = fleurs[0]["audio"]
    print(audio)

    # Lendo áudio
    wav, sr = torchaudio.load("JN.wav")
    wav = wav.unsqueeze(0).cuda()
    
    # torchaudio.nn.functional.resample
    
    # Load checkpoint (either hubert_soft or hubert_discrete)
    hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True).cuda()

    from torch.cuda.amp import autocast
    with autocast():
        #units = hubert.units(wav)
        units = hubert.units(audio)

    print(units.shape)
    prompts = units
    prompts_tokens = [units]

    results = generator.text_completion(
        prompts_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        #print(f"> {generator.tokenizer.decode(result['generation'])}")
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
