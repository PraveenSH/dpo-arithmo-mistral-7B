from datasets import Dataset, concatenate_datasets, load_dataset
from bs4 import BeautifulSoup
import html
import json
import re


def detect_math_problem(question):

    math_pattern = r'\bcalculate\b'
    match = re.search(math_pattern, question.lower())
    
    if match:
        return True
    else:
        return False


def html_to_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    text = html.unescape(text)
    
    return text


def extract_data():
    data_dir = "data/math.stackexchange.com"
    dataset = load_dataset(
            "HuggingFaceH4/pmp-stack-exchange",
            data_dir=data_dir,
            split="train",
            streaming=True,
        )

    output_data = get_preference_samples(dataset)

    with open("Data/math_stack_exchange_dpo_reduced.json", 'w') as fl:
        json.dump(output_data, fl, indent=4)


def get_preference_samples(dataset):
    output_data = []
    for i, d in enumerate(iter(dataset)):
        question = d["question"]
        answers = d["answers"]
        num_answers = len(answers)

        if num_answers < 2:
            continue

        min_score = answers[0]["pm_score"]
        max_score = answers[0]["pm_score"]
        min_answer = answers[0]
        max_answer = answers[0]

        for ans in answers:
            if ans["pm_score"] <= min_score:
                min_score = ans["pm_score"]
                min_answer = ans
            if ans["pm_score"] >= max_score:
                max_score = ans["pm_score"]
                max_answer = ans

        if min_score == max_score:
            continue

        processed_question = html_to_text(question)
        if detect_math_problem(processed_question):
            data_json = {"question": processed_question,
                         "chosen": html_to_text(max_answer['text']),
                         "rejected": html_to_text(min_answer['text'])
                         }

            output_data.append(data_json)
    return output_data


if __name__ == "__main__":
    extract_data()
