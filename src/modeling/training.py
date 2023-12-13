from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig
from trl import DPOTrainer


def load_checkpoints():
    model_id = "akjindal53244/Arithmo-Mistral-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, torch_dtype=torch.float16,
                                                 load_in_4bit=True)
    ref_model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, torch_dtype=torch.float16,
                                                     load_in_4bit=True)
    return tokenizer, model, ref_model


def dpo_format(example):
    return {
        "prompt": example['question'],
        "chosen": example['chosen'],
        "rejected": example['rejected'],
    }


def load_custom_dataset(file_path):
    # Load JSON data using load_dataset
    dataset = load_dataset('json', data_files=file_path)
    return dataset


def train():
    tokenizer, model, ref_model = load_checkpoints()

    file_path = 'math_stack_exchange_dpo.json'
    custom_dataset = load_custom_dataset(file_path)
    dataset = custom_dataset.map(
        dpo_format
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=2000,
        save_strategy="no",
        logging_steps=1,
        output_dir="output_di",
        warmup_steps=100,
        fp16=True,
    )
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,
        max_prompt_length=512,
    )

    dpo_trainer.train()
    save_model(dpo_trainer, tokenizer)


def save_model(dpo_trainer, tokenizer):
    dpo_trainer.model.save_pretrained("arithmo_dpo_checkpoint")
    tokenizer.save_pretrained("arithmo_dpo_checkpoint")


if __name__ == "__main__":
    train()
