import wandb
import random
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

# TO DO add arguments
params = {
    'num_epochs': 8,
    'batch_size': 7,
    'lr': 2e-5,
    'deday': 0.01
}

def main():

    # TO DO: add arguments

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="llama-compare",
        entity="dimap-ai",
    
        # track hyperparameters and run metadata
        config={
            "learning_rate": 2e-5,
            "architecture": "Transformer",
            "dataset": "CNN News",
            "epochs": 4,
        }
    )

    # check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # load data
    cnn_news_ptbr = load_dataset('celsowm/cnn_news_ptbr', split="train")
    print(cnn_news_ptbr)

    # remove columns
    dataset = load_dataset('celsowm/cnn_news_ptbr', split="train")
    columns_to_remove = ['titulo', 'link', 'resumo', 'categoria', 'data_hora']
    dataset = dataset.remove_columns(columns_to_remove)

    ntrain = len(dataset)
    print(f'Number of training sentences: {ntrain}')

    ind1 = random.randint(0, ntrain-1)
    print('\nTraining example: \n\n', dataset[ind1])

    # Model from Hugging Face hub
    # the model is already downloaded, change this!
    base_model = "NousResearch/Llama-2-7b-hf"

    # Fine-tuned model
    new_model = "llama-2-7b-news-gen"

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # tokenize dataset so that llama input can be List[List[int]]

    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0}
    )

    # Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
    model = prepare_model_for_kbit_training(model)

    # Set training arguments
    training_arguments = TrainingArguments(
        output_dir=new_model,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=25,
        logging_steps=10,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        report_to="wandb",
        run_name=new_model
#        max_steps=2, # Remove this line for a real fine-tuning
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="texto",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model)
    
    trainer.push_to_hub("fernandals/llama-2-7b-news-gen")


if __name__ == "__main__":
    main()
