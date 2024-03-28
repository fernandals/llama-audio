# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
import os
import numpy as np

from llama import Llama
from typing import List

PATH = 'audios/dac/'

def get_dacs(dir_path: str = PATH) -> [str]:
    codes_path = list(map(lambda x: PATH + '/' + x, os.listdir(PATH)))
    return codes_path

def get_code_from_path(path: str) -> torch:
    code = np.load(path, allow_pickle=True)[()]
    code = torch.from_numpy(code['codes'].astype(int))
    return code

def get_codes_from_dir(dir_path: str = PATH):
    dacs = get_dacs(dir_path)
    codes = list(map(lambda x: get_code_from_path(x), dacs))
    return codes

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
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

    # codes from audio!
    codes = get_codes_from_dir()
    print(codes)

    prompts: List[List[int]] = codes
   
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
