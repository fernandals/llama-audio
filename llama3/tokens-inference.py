# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

# script to infere on llama3 using audio tokens

from typing import List

import fire

from llama import Llama

from datasets import load_dataset

from torch.cuda.amp import autocast

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
    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    fleurs = load_dataset("google/fleurs", "en_us", split="test")
    id_speaker = fleurs[1]["id"]
    transcript = fleurs[1]["transcription"]
    audio = fleurs[1]["audio"]

    text_prompts: List[str] = [transcript]
    text_prompt_tokens = [generator.tokenizer.encode(x, bos=True, eos=True) for x in prompts]
   
    hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True).cuda()
    with autocast():
        units = hubert.units(audio)

    audio_prompt_tokens = [units] 

    prompts_tokens = id_speaker + text_prompt_tokens + audio_prompt_tokens

    results = generator.text_completion(
        prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompt_tokens, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
