import hashlib
import json
import os
import re
from typing import List, Dict

import ftfy
from jsonlines import jsonlines


def read_json(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        instance = json.load(file)
    return instance


def read_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def write_json(instance: Dict, file_path: str):
    with open(file_path, "w") as file:
        json.dump(instance, file)


def write_jsonl(instances: List[Dict], file_path: str):
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


def get_pid_for_title_paragraph_text(title: str, paragraph_text: str) -> str:
    title = ftfy.fix_text(title.strip())
    paragraph_text = ftfy.fix_text(paragraph_text.strip())

    if paragraph_text.startswith("Wikipedia Title: " + title + "\n"):
        paragraph_text = paragraph_text.replace("Wikipedia Title: " + title + "\n", "").strip()

    if paragraph_text.startswith("Wikipedia Title: " + title + " \n"):
        paragraph_text = paragraph_text.replace("Wikipedia Title: " + title + " \n", "").strip()

    if paragraph_text.startswith("Title: " + title + "\n"):
        paragraph_text = paragraph_text.replace("Title: " + title + "\n", "").strip()

    if paragraph_text.startswith("Title: " + title + " \n"):
        paragraph_text = paragraph_text.replace("Title: " + title + " \n", "").strip()

    title = "".join([i if ord(i) < 128 else " " for i in title]).lower()
    paragraph_text = "".join([i if ord(i) < 128 else " " for i in paragraph_text]).lower()

    title = re.sub(r" +", " ", title)
    paragraph_text = re.sub(r" +", " ", paragraph_text)

    # NOTE: This is more robust, but was done after V2 big exploration.
    # So uncomment it for rerunning evals for those experiments.
    title = re.sub(r" +", "", title)
    paragraph_text = re.sub(r" +", "", paragraph_text)

    pid = "___".join(
        [
            "pid",
            hashlib.md5(title.encode("utf-8")).hexdigest(),
            hashlib.md5(paragraph_text.encode("utf-8")).hexdigest(),
        ]
    )

    return pid


def format_drop_answer(answer_json):
    if answer_json["number"]:
        return answer_json["number"]
    if len(answer_json["spans"]):
        return answer_json["spans"]
    # only date possible
    date_json = answer_json["date"]
    if not (date_json["day"] or date_json["month"] or date_json["year"]):
        print("Number, Span or Date not set in {}".format(answer_json))
        return None
    return date_json["day"] + "-" + date_json["month"] + "-" + date_json["year"]


def clean_query(query: str):
    # TODO Due to weird processing habit of DeepSeek
    if "</think>" in query:
        index = query.find('</think>')
        query = query[index + 8:].strip()
    if "**Answer" in query:
        index = query.find('**Answer"')
        query = query[:index].strip()
    if "Step-by-step explanation" in query:
        index = query.find('Step-by-step explanation"')
        query = query[:index].strip()
    return query


def load_ground_truths(
        ground_truth_file_path: str,
) -> Dict:
    id_to_ground_truths = {}
    with jsonlines.open(ground_truth_file_path, 'r') as input_file:
        for line in input_file:
            # import pdb; pdb.set_trace()
            qid = line['question_id']
            answer = line['answers_objects'][0]['spans']
            id_to_ground_truths[qid] = answer
    return id_to_ground_truths


def parse_answer(item: Dict):
    answer = item['answers_objects'][0]['spans']
    return answer


def get_qa_dataset_path():
    result = os.getenv("QA_DATASET_DIR")
    assert result, "QA_DATASET_DIR environment variable not set"
    return result
