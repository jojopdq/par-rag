import copy
import json
import os
from collections import Counter
from typing import Dict

from tqdm import tqdm

from src.lib import format_drop_answer, get_pid_for_title_paragraph_text, read_jsonl, get_qa_dataset_path


class DatasetLoader:
    def __init__(self, env):
        self.root_path = os.path.join(get_qa_dataset_path(), "processed_data")
        self.env = env

    def load(self, corpus_name: str):
        result = []
        file_path = f"{self.root_path}/{corpus_name}/{self.env}_subsampled.jsonl"
        items = read_jsonl(file_path)
        for item in items:
            _id = item['question_id']
            question = item['question_text']
            answers_objects = item["answers_objects"]
            ground_truth = answers_objects[0]["spans"]
            result.append({'id': _id, 'question': question, 'ground_truth': ground_truth})
        return result

    def id_name(self, corpus_name: str):
        if corpus_name == 'musique':
            return 'id'
        else:
            return '_id'

    def read_examples(self, file):
        with open(file, "r") as input_fp:
            for line in tqdm(input_fp):
                if not line.strip():
                    continue
                input_instance = json.loads(line)
                qid = input_instance["question_id"]
                query = question = input_instance["question_text"]
                answers_objects = input_instance["answers_objects"]

                formatted_answers = [  # List of potentially validated answers. Usually it's a list of one item.
                    tuple(format_drop_answer(answers_object)) for answers_object in answers_objects
                ]
                answer = Counter(formatted_answers).most_common()[0][0]

                output_instance = {
                    "qid": qid,
                    "query": query,
                    "answer": answer,
                    "question": question,
                }

                for paragraph in input_instance.get("pinned_contexts", []) + input_instance["contexts"]:
                    assert not paragraph["paragraph_text"].strip().startswith("Title: ")
                    assert not paragraph["paragraph_text"].strip().startswith("Wikipedia Title: ")

                title_paragraph_tuples = []

                output_instance["titles"] = [e[0] for e in title_paragraph_tuples]
                output_instance["paras"] = [e[1] for e in title_paragraph_tuples]

                pids = [
                    get_pid_for_title_paragraph_text(title, paragraph_text)
                    for title, paragraph_text in zip(output_instance["titles"], output_instance["paras"])
                ]
                output_instance["pids"] = pids

                output_instance["real_pids"] = [
                    paragraph["id"]
                    for paragraph in input_instance["contexts"]
                    if paragraph["is_supporting"] and "id" in paragraph
                ]

                for para in output_instance["paras"]:
                    assert not para.strip().startswith("Title: ")
                    assert not para.strip().startswith("Wikipedia Title: ")

                # Backup Paras and Titles are set so that we can filter from the original set
                # of paragraphs again and again.
                if "paras" in output_instance:
                    output_instance["backup_paras"] = copy.deepcopy(output_instance["paras"])
                    output_instance["backup_titles"] = copy.deepcopy(output_instance["titles"])

                if "valid_titles" in input_instance:
                    output_instance["valid_titles"] = input_instance["valid_titles"]

                output_instance["metadata"] = {}
                output_instance["metadata"]["level"] = input_instance.get("level", None)
                output_instance["metadata"]["type"] = input_instance.get("type", None)
                output_instance["metadata"]["answer_type"] = input_instance.get("answer_type", None)
                output_instance["metadata"]["simplified_answer_type"] = input_instance.get(
                    "simplified_answer_type", None
                )

                output_instance["metadata"]["gold_titles"] = []
                output_instance["metadata"]["gold_paras"] = []
                output_instance["metadata"]["gold_ids"] = []
                for paragraph in input_instance["contexts"]:
                    if not paragraph["is_supporting"]:
                        continue
                    title = paragraph["title"]
                    paragraph_text = paragraph["paragraph_text"]
                    output_instance["metadata"]["gold_titles"].append(title)
                    output_instance["metadata"]["gold_paras"].append(paragraph_text)
                    output_instance["metadata"]["gold_ids"].append(paragraph.get("id", None))

                yield output_instance

    def load_ground_truths(self, ground_truth_file_path: str, ) -> Dict:
        id_to_ground_truths = {}
        for example in self.read_examples(ground_truth_file_path):
            id_ = example["qid"]
            id_to_ground_truths[id_] = example["answer"]
        return id_to_ground_truths

    def load_predictions(self, prediction_file_path: str) -> Dict:
        with open(prediction_file_path, "r") as file:
            id_to_predictions = json.load(file)
        return id_to_predictions
