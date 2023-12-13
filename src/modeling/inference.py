import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import json
from peft import PeftModel
from datasets import load_from_disk


model_path = "Arithmo-Mistral-7B"
check_point = "arithmo_dpo_checkpoint"

run_model_on_gpu = True

use_4bit = True
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False


base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    return_dict=True,
    torch_dtype=torch.float16,
).to(torch.device('cuda:0'))

# Merge base model with the adapter
model = PeftModel.from_pretrained(base_model, check_point)
model = model.merge_and_unload()
model.to(torch.device('cuda:0'))


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


gsm8k_test = load_from_disk('gsm8k')
dataset_size = len(gsm8k_test['test'])
print(f"gsm8k_test size: {dataset_size}")

data = gsm8k_test['test']

output_file = open("gsm8k_test_fresh.json", 'w')

output_data = []
for i in range(len(data)):
    input_text = data[i]['question']
    actual_answer = data[i]['answer']
    
    input_text_ft = f"Question: {input_text}\n\nAnswer:"

    if run_model_on_gpu:
        inputs_ft = tokenizer(input_text_ft, return_tensors="pt", padding=True).to("cuda")
    else:
        inputs_ft = tokenizer(input_text_ft, return_tensors="pt", padding=True)
    
    generated_ids = model.generate(**inputs_ft, max_new_tokens=1024, temperature=0.0)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    data_json = {"question": input_text,
                 "ground_truth": actual_answer,
                 "prediction": output
                 }

    output_data.append(data_json)

json.dump(output_data, output_file, indent=4)
