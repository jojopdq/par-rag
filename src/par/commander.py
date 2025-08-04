import copy
import json
import os
import pprint
import re
import time
from typing import List, Dict

from loguru import logger
from tqdm import tqdm

from src.bert_classifier_plus import QuestionComplexityClassifier
from src.langchain_util import init_langchain_model
from src.lib import load_ground_truths, get_qa_dataset_path
from src.par.dataset_loader import DatasetLoader
from src.par.director import Director
from src.par.generator import Generator
from src.par.planner import Planner
from src.par.step_executor import StepExecutor
from src.par.supervisor import Supervisor
from src.retriever import Colbertv2Retriever

PLAN_MODE = 'no_plan_mode'
REVIEW_MODE = 'no_review_mode'
NO_EXAMPLE_MODE = None
RANDOM_EXAMPLE_MODE = 'random'
SIMILAR_EXAMPLE_MODE = 'similar'


class Commander:
    def __init__(self, config: Dict):
        self.config = config
        self.client = init_langchain_model(llm=config["current_llm_provider"], model_name=config["current_llm_model"],
                                           config=config)

        self.env = self.config['env']
        self.ablation_mode = config.get("ablation_mode", None)
        self.example_mode = config.get("example_mode", NO_EXAMPLE_MODE)
        self.round_name = config.get("round_name", None)
        model_path = config.get('classifier', {}).get('model_path')
        assert model_path is not None, "the model path of the classifier not found"
        classifier = QuestionComplexityClassifier(model_path)

        self.dataset_loader = DatasetLoader(self.env)
        self.planner = Planner(self.client)
        self.step_executor = StepExecutor(self.client)
        self.supervisor = Supervisor(self.client)
        self.generator = Generator(self.client)
        self.director = Director(config, classifier)

        self.retriever = None

        self.target = 'par-rag'

    def execute(self, corpus_name: str):
        items = self.dataset_loader.load(corpus_name)
        result = {}
        for item in tqdm(items):
            start_time = time.time()
            question_id = item['id']
            question = item['question']
            try:
                prediction, total_tokens, status, supervised_status = self.process(corpus_name, question)
                total_time = time.time() - start_time
                record = {'question': question, 'answer': prediction, 'total_time': total_time,
                          'total_tokens': total_tokens,
                          'status': status,
                          'supervised_status': supervised_status
                          }
                result[question_id] = record
            except Exception as e:
                logger.debug(f"Exception: {e} occurred when processing question: {item}")
                total_time = time.time() - start_time
                record = {'question': question, 'answer': str(e), 'total_time': total_time, 'total_tokens': 0,
                          'status': 'FAILED'}
                result[question_id] = record
        ground_truths = load_ground_truths(
            f'{get_qa_dataset_path()}/processed_data/{corpus_name}/{self.env}_subsampled.jsonl')
        for question_id, record in result.items():
            ground_truth = ground_truths[question_id]
            record['ground_truth'] = ground_truth
        output_path = f'output/{self.target}/{self.round_name}'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = f'{output_path}/{corpus_name}_{self.env}.json'
        json_object = json.dumps(result, indent=4, ensure_ascii=False)
        with open(output_file, "w", encoding='UTF-8') as outfile:
            outfile.write(json_object)
        return result

    def execute_only_one(self, corpus_name, query: str):
        try:
            result, _, _, _ = self.process(corpus_name, query)
            return result
        except Exception as e:
            logger.debug(f"Exception occurred when processing question {query}: {e}")
            return ""

    def process(self, corpus_name, question: str):
        pp = pprint.PrettyPrinter(indent=2, width=160, sort_dicts=False)
        total_tokens = 0
        consumed_tokens = 0
        supervised_status = None
        logger.info(
            f"=========corpus name:{corpus_name}, question:{question}, ablation mode:{self.ablation_mode}, example mode:{self.example_mode}=========")
        logger.info("------------Plan------------")
        if self.ablation_mode == PLAN_MODE:
            plan = [{'Step 1': {'Thought': '', 'Question': question, 'Action': 'Retrieve'}}]
        else:
            history_record = None
            if self.example_mode == SIMILAR_EXAMPLE_MODE:
                complexity_score, history_record = self.director.fetch_most_similar_history_record(question)
                logger.info(
                    f"current question got a complexity score:{complexity_score}, fetched most similar history record: \n{history_record}")
            elif self.example_mode == RANDOM_EXAMPLE_MODE:
                complexity_score, history_record = self.director.fetch_random_history_record(question)
            # Create plan for question
            plan, consumed_tokens = self.planner.create(question, history_record)
        total_tokens += consumed_tokens
        logger.info(
            f"plan: \n{pp.pformat(plan)}\n, total tokens:{total_tokens}")
        if not plan:
            raise Exception(f"Could not create plan for question {question}")
        steps = copy.deepcopy(plan)
        num = 1
        trajectory_records = []
        status = None
        for step in steps:
            logger.info(f"[Step {num}]: {step}")
            current_step_no = f"Step {num}"
            current_step = step.get(current_step_no, {})
            current_question = current_step.get("Question", "")
            logger.info("------------Act------------")
            retrieved_passages = self.retrieve(corpus_name, current_question)
            samples = self.construct_samples(retrieved_passages)
            answer, consumed_tokens = self.step_executor.answer(current_question, samples)
            total_tokens += consumed_tokens
            evidences = self.select_cited_evidences(answer, retrieved_passages)

            logger.info(
                f"current question:{current_question}, answer:{answer}, total tokens:{total_tokens}, samples:\n{pp.pformat(samples)}")
            if not self.ablation_mode or (self.ablation_mode and self.ablation_mode != REVIEW_MODE):
                logger.info("------------Review------------")
                evaluation_result, consumed_tokens = self.supervisor.evaluate(current_question, answer, evidences)
                total_tokens += consumed_tokens
                logger.info(f"evaluation result: {evaluation_result}, total tokens: {total_tokens}")
                evaluation_status = evaluation_result['Status']
                if evaluation_status != 'PASS':
                    if "I don't know" not in answer:
                        another_retrieved_passages = self.retrieve(corpus_name, answer)
                    else:
                        another_retrieved_passages = retrieved_passages
                    samples = self.construct_samples(another_retrieved_passages)
                    revised_content, consumed_tokens = self.supervisor.rectify(current_question, samples)
                    total_tokens += consumed_tokens
                    supervised_status = revised_content.get('Status')
                    if supervised_status == 'UNCONFIDENT':
                        status = 'UNCONFIDENT'
                    elif supervised_status == 'REVISED':
                        status = 'REVISED'
                        supervised_status = 'REVISED'
                        retrieved_passages = another_retrieved_passages
                        answer = revised_content.get('Answer')
                        evidences = self.select_cited_evidences(answer, retrieved_passages)
                    logger.info(
                        f"review result: {revised_content}, supervised status:{supervised_status}, total tokens: {total_tokens}, samples:\n{pp.pformat(samples)}")
                else:
                    status = 'PASS'
            current_step["Answer"] = answer
            trajectory_records.append(
                {'Step': num, 'Question': current_question, 'Thought': current_step.get('Thought'), 'Answer': answer,
                 'Evidences': evidences, 'Status': status})

            num += 1
            if num > len(steps):
                break
            next_step_no = f"Step {num}"
            print(f"num: {num}, next_step_no: {next_step_no}, len(steps): {len(steps)})")
            next_step = steps[num - 1].get(next_step_no, {})
            next_question = next_step.get("Question", None)
            if not next_question:
                raise RuntimeError(f"Could not find next question in {next_step}")
            trajectories = self.to_trajectories(trajectory_records)
            refined_next_question, consumed_tokens = self.step_executor.refine_next_question(next_question,
                                                                                             pp.pformat(
                                                                                                 trajectories))
            total_tokens += consumed_tokens
            logger.info(f"next question: {next_question}")
            logger.info(f"after refined: {refined_next_question}, total tokens: {total_tokens}")
            next_step["Question"] = refined_next_question

        trajectories = self.to_trajectories(trajectory_records)
        logger.info(f"[Final Step]: answer the question: {question}, trajectories:\n{pp.pformat(trajectories)}")
        predicted_answer, consumed_tokens = self.generator.read(question, pp.pformat(trajectories))
        total_tokens += consumed_tokens

        evidences = []
        for trajectory_record in trajectory_records:
            evidences.extend(trajectory_record['Evidences'])
        if self.supervisor.is_correct(question, predicted_answer, evidences):
            status = 'PASS'
        else:
            status = 'UNCONFIDENT'
        logger.info(
            f"answer:{predicted_answer}, status:{status}, supervised_status:{supervised_status},total_tokens:{total_tokens}")
        return predicted_answer, total_tokens, status, supervised_status

    def construct_samples(self, retrieved_passages: List) -> List:
        samples = []
        idx = 1
        for retrieved_context in retrieved_passages:
            args = retrieved_context.split("|")
            if len(args) == 2:
                samples.append(f"[Source {idx}]: {args[1]}|{args[0]}")
            else:
                samples.append(f"[Source {idx}]: {retrieved_context}")
            idx += 1
        return samples

    def retrieve(self, corpus_name, query):
        if not self.retriever:
            self.retriever = Colbertv2Retriever(self.config)
        return self.retriever.fetch(query)

    def to_trajectories(self, trajectories):
        result = []
        for item in trajectories:
            result.append(
                {'Step': item['Step'], 'Question': item['Question'], 'Thought': item['Thought'],
                 'Answer': item['Answer'], 'Evidences': item['Evidences']})
        return result

    def select_cited_evidences(self, answer, retrieved_passages):
        temp = []
        result = re.findall(r'\[(\d+)\]', answer)  #
        for item in result:
            item = int(item.replace('[', '').replace(']', '').strip())
            if item >= len(retrieved_passages):
                continue
            source = retrieved_passages[item - 1]
            temp.append(source)
        sources = self.construct_samples(temp)
        return sources
