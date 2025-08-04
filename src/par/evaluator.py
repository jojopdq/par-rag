import os
import re
import string
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_util import LangChainModel
from src.lib import read_json, write_json, write_jsonl, load_ground_truths, get_qa_dataset_path
from src.metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric
from src.par.dataset_loader import DatasetLoader
from src.qa import hotpotqa_evaluation, twowikimultihopqa_evaluation, musique_evaluation
from src.qa.musique_evaluation import read_jsonl


class Evaluator:
    def __init__(self, client: LangChainModel, env, target, round_name=None):
        self.root_path = os.getenv("QA_DATASET_DIR", get_qa_dataset_path())
        self.tmp_path = "/tmp"
        self.dataset_loader = DatasetLoader(env)
        self.client = client
        self.env = env
        self.target = target
        self.round_name = round_name

    def process(self, corpus_name: str):
        ground_truths_file_path = f"{self.root_path}/processed_data/{corpus_name}/{self.env}_subsampled.jsonl"
        id_to_ground_truths = self.dataset_loader.load_ground_truths(
            ground_truths_file_path,
        )
        # prep predictions
        prediction_file_path = f"output/{self.target}/{self.round_name}/{corpus_name}_{self.env}.json"
        prediction_results = self.dataset_loader.load_predictions(prediction_file_path)
        prediction_results = {
            id_: prediction for id_, prediction in prediction_results.items() if id_ in id_to_ground_truths.keys()
        }

        total_time = 0
        total_tokens = 0
        id_to_predictions = {}
        for id_, item in prediction_results.items():
            answer = item.get('answer')
            total_time += item.get('total_time', 0)
            total_tokens += item.get('total_tokens', 0)
            id_to_predictions[id_] = answer

        average_time = round(total_time / len(prediction_results), 2)
        average_tokens = total_tokens / len(id_to_predictions)
        question_ids = list(id_to_predictions.keys())
        for id_, prediction in id_to_predictions.items():
            if isinstance(prediction, list) and len(prediction) == 1:
                id_to_predictions[id_] = str(prediction[0])
            elif isinstance(prediction, list) and len(prediction) > 1:
                id_to_predictions[id_] = " ".join([str(e) for e in prediction])
                print("WARNING: Found a list answer prediction, concatenating it.")

        # os.makedirs(".temp", exist_ok=True)

        perf_metrics = {'average_time': average_time, 'average_tokens': average_tokens}
        if corpus_name == "hotpotqa":
            result = self.__eval_hotpotqa(question_ids, id_to_predictions)
        elif corpus_name == "2wikimultihopqa":
            result = self.__eval_2wikimultihopqa(question_ids, id_to_predictions)
        elif corpus_name == "musique":
            result = self.__eval_musique(question_ids, id_to_predictions)
        elif corpus_name in ['nq', 'squad', 'trivia']:
            result = self.eval_single_step_dataset(corpus_name, question_ids, id_to_predictions)
        else:
            raise NotImplementedError

        return {**perf_metrics, **result}

    def __eval_hotpotqa(self, question_ids=None, id_to_predictions=None):
        # prepare ground_truth file:
        temp_ground_truth_file_path = os.path.join(self.tmp_path, uuid.uuid4().hex)
        original_data = read_json(os.path.join(self.root_path, "raw_data", "hotpotqa", "hotpot_dev_distractor_v1.json"))
        filtered_data = [datum for datum in original_data if datum["_id"] in question_ids]
        write_json(filtered_data, temp_ground_truth_file_path)

        # prepare prediction file:
        temp_prediction_file_path = os.path.join(self.tmp_path, uuid.uuid4().hex)
        for prediction in id_to_predictions.values():
            if not isinstance(prediction, str):
                print("WARNING: Found an answer prediction that's not a string.")

        data = {
            "answer": {id_: str(prediction) for id_, prediction in id_to_predictions.items()},
            "sp": {id_: [["", 0]] for id_, _ in id_to_predictions.items()},
        }
        write_json(data, temp_prediction_file_path)

        total_acc = 0
        for item in filtered_data:
            actual_output = id_to_predictions[item["_id"]]
            expected_output = item["answer"]
            total_acc += self.calculate_accuracy(actual_output, expected_output)

        metrics_ = hotpotqa_evaluation.eval(temp_prediction_file_path, temp_ground_truth_file_path)
        metrics = {
            "f1": round(metrics_["f1"], 3),
            "em": round(metrics_["em"], 3),
            "precision": round(metrics_["prec"], 3),
            "recall": round(metrics_["recall"], 3),
            "count": len(id_to_predictions),
            "acc": round(total_acc / len(id_to_predictions), 3),
        }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)

        return metrics

    def __eval_2wikimultihopqa(self, question_ids=None, id_to_predictions=None):
        # prepare ground_truth file:
        temp_ground_truth_file_path = os.path.join(self.tmp_path, uuid.uuid4().hex)
        original_data = read_json(os.path.join(self.root_path, "raw_data", "2wikimultihopqa", "dev.json"))
        filtered_data = [datum for datum in original_data if datum["_id"] in question_ids]
        write_json(filtered_data, temp_ground_truth_file_path)

        # prepare prediction file:
        temp_prediction_file_path = os.path.join(self.tmp_path, uuid.uuid4().hex)
        for prediction in id_to_predictions.values():
            if not isinstance(prediction, str):
                print("WARNING: Found an answer prediction that's not a string.")

        data = {
            "answer": {id_: str(prediction) for id_, prediction in id_to_predictions.items()},
            "sp": {id_: [["", 0]] for id_, _ in id_to_predictions.items()},
            "evidence": {id_: ["", "", ""] for id_, _ in id_to_predictions.items()},
        }
        write_json(data, temp_prediction_file_path)

        alias_file_path = os.path.join(self.root_path, "raw_data", "2wikimultihopqa", "id_aliases.json")
        metrics_ = twowikimultihopqa_evaluation.eval(temp_prediction_file_path, temp_ground_truth_file_path,
                                                     alias_file_path)

        total_acc = 0
        for item in filtered_data:
            actual_output = id_to_predictions[item["_id"]]
            expected_output = item["answer"]
            total_acc += self.calculate_accuracy(actual_output, expected_output)

        metrics = {
            "f1": round(metrics_["f1"] / 100, 3),
            "em": round(metrics_["em"] / 100, 3),
            "precision": round(metrics_["prec"] / 100, 3),
            "recall": round(metrics_["recall"] / 100, 3),
            "count": len(id_to_predictions),
            "acc": round(total_acc / len(id_to_predictions), 3),
        }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)

        return metrics

    def __eval_musique(self, question_ids=None, id_to_predictions=None):
        # prepare ground_truth file:
        temp_ground_truth_file_path = os.path.join(self.tmp_path, uuid.uuid4().hex)
        original_data = read_jsonl(os.path.join(self.root_path, "raw_data", "musique", "musique_ans_v1.0_dev.jsonl"))
        original_keyed_data = {datum["id"]: datum for datum in original_data}
        filtered_data = [original_keyed_data[qid] for qid in question_ids]
        write_jsonl(filtered_data, temp_ground_truth_file_path)

        # prepare prediction file:
        temp_prediction_file_path = os.path.join(self.tmp_path, uuid.uuid4().hex)
        for prediction in id_to_predictions.values():
            if not isinstance(prediction, str):
                print("WARNING: Found an answer prediction that's not a string.")

        data = [
            {
                "id": id_,
                "predicted_answer": str(id_to_predictions[id_]),
                "predicted_support_idxs": [0, 1],
                "predicted_answerable": True,
            }
            for id_ in question_ids
        ]
        write_jsonl(data, temp_prediction_file_path)

        total_acc = 0
        for item in filtered_data:
            actual_output = id_to_predictions[item["id"]]
            expected_output = item["answer"]
            total_acc += self.calculate_accuracy(actual_output, expected_output)

        metrics_ = musique_evaluation.evaluate(temp_prediction_file_path, temp_ground_truth_file_path)
        metrics = {
            "f1": round(metrics_["answer_f1"], 3),
            "em": round(metrics_["answer_em"], 3) if "answer_em" in metrics_ else None,
            "count": len(id_to_predictions),
            "acc": round(total_acc / len(id_to_predictions), 3),
        }

        os.remove(temp_ground_truth_file_path)
        os.remove(temp_prediction_file_path)

        return metrics

    def eval_single_step_dataset(self, dataset_name, question_ids=None, id_to_predictions=None):
        metrics = [SquadAnswerEmF1Metric()]
        # prepare ground_truth file:
        original_ground_truths_mapping = load_ground_truths(
            self.root_path + "/processed_data/" + dataset_name + f'/{self.env}_subsampled.jsonl')
        id_to_ground_truths = {}
        for qid in question_ids:
            if qid in original_ground_truths_mapping:
                id_to_ground_truths[qid] = original_ground_truths_mapping[qid]

        total_acc = 0

        for id_ in set(id_to_ground_truths.keys()):
            ground_truth = id_to_ground_truths[id_]
            prediction = id_to_predictions[id_]

            assert isinstance(prediction, (str, list))
            if isinstance(prediction, str):
                if prediction.strip().startswith("[") or prediction.strip().endswith("]"):
                    prediction = [e for e in prediction.replace('"', "").replace("[", "").replace("]", "").split(",")]
                else:
                    prediction = [prediction]

            assert isinstance(prediction, (list, tuple))
            prediction = [str(e) for e in prediction]
            prediction = [answer_extractor(_prediction) for _prediction in prediction]  # Temporary.
            actual_output = prediction[0]

            acc = 0
            if actual_output != "I don't know":
                expected_outputs = [normalize_answer(i) for i in ground_truth]
                for expected_output in expected_outputs:
                    acc = self.calculate_accuracy(normalize_answer(actual_output), expected_output)
                    if acc == 1:
                        break
            total_acc = total_acc + acc
            metrics[0](prediction[0], ground_truth)

        total_acc = total_acc / len(id_to_predictions)
        evaluation_results = metrics[0].get_metric()
        evaluation_results['acc'] = total_acc

        return evaluation_results

    def calculate_accuracy(self, actual_output, expected_output):
        if actual_output == expected_output:
            return 1
        instruction = f"""
        You are a strict evaluator. Compare the actual output and the expected output to determine if they are semantically equivalent—that is, if they express the same meaning, even if phrased differently.
        Return 1 if the two outputs are semantically equivalent.
        Return 0 if they are not equivalent in meaning.
        Respond with a single number: 1 or 0.
        Expected Output: {expected_output}
        Actual Output: {actual_output}
        Are the two outputs semantically equivalent? 
        """
        messages = [HumanMessage(instruction)]
        try:
            messages = ChatPromptTemplate.from_messages(messages).format_prompt()
            chat_completion = self.client.invoke(messages.to_messages())
            return int(chat_completion.content)
        except Exception as e:
            print(e)
            return 0


def answer_extractor(potentially_cot: str) -> str:
    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]

    cot_regex = re.compile(".* answer is:? (.*)\\.?")
    match = cot_regex.match(potentially_cot)
    if match:
        output = match.group(1)
        if output.endswith("."):
            output = output[:-1]
    else:
        output = potentially_cot

    return output


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
