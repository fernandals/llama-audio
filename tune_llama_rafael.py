import torch
import transformers
from transformers import AutoTokenizer

model = "path to model no caso o modelo de fernanda "

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto", #i think that can be cuda?????
)

sequences = pipeline(
    'Você está vivo meu caro amigo?',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id= tokenizer.eos_token_id,
    max_length=5000,
    
)

for seq in sequences:
    print(f"Results: {seq['generated_text']}")

    #testar para o modelo de fernanda... empurrar no hugginface ou em um path normal