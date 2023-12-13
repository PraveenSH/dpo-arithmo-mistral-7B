# dpo-arithmo-mistral-7B
Aligning LLM with Direct Preference Optimization and LoRA to improve the mathematical reasoning capabilities

## Introduction
[Arithmo-Mistral-7B](https://github.com/akjindal53244/Arithmo-Mistral-7B) was trained to reason and answer mathematical problems. The model was trained with [Arithmo-data](https://huggingface.co/datasets/akjindal53244/Arithmo-Data).

In this work, we further align the model using preference dataset and direct preference optimization method [DPO](https://arxiv.org/abs/2305.18290).

## Dataset
The preference dataset is derived from the [stack exchange dataset](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences) which contains questions and answers from the Stack Overflow Data Dump for the purpose of preference model training. This contains questions and answers for various topics. For this work, we used only question and answers from [math.stackexchange.com](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences/tree/main/data/math.meta.stackexchange.com) sub-folder.

The questions are grouped with answers that are assigned a score corresponding to the Anthropic paper:
```
score = log2 (1 + upvotes) rounded to the nearest integer, plus 1 if the answer was accepted by the questioner (we assign a score of −1 if the number of upvotes is negative).
```

We performed following processing to derive the final dataset.
1) Basic pre-processing ([code](https://github.com/PraveenSH/dpo-arithmo-mistral-7B/blob/main/src/data_processing/stack_exchange_data.py)) to clean the text
2) Filter Mathematical question using regex based detector ([code](https://github.com/PraveenSH/dpo-arithmo-mistral-7B/blob/main/src/data_processing/stack_exchange_data.py))
3) For each question, extract 2 answers - one with highest score and one with the lowest score. Former is used as Preferred response and latter is used as the rejected response
4) The final dataset contains ~18k samples and the dataset can be found here - [hugginf-face link]()

## Model training.
1) The model is initialized with the [Arithmo-Mistral-7B](https://huggingface.co/akjindal53244/Arithmo-Mistral-7B) checkpoint
2) It is further trained using [dpo trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) with LoRA
3) The final checkpoint is released here [hugging face link]()

## Model evaluation

## Results
