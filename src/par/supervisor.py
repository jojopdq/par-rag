import json
import pprint
from typing import List

import numpy as np
from easydict import EasyDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from loguru import logger


class Supervisor:
    """
    Review Module.
    """

    def __init__(self, client):
        self.client = client
        self.model = pipeline("text2text-generation", "osunlp/attrscore-flan-t5-xl")
        self.ratio = 0.5
        self.threshold = 0.75
        self.instruction = '''
        You are an expert with rigorous logic and good reasoning. 
        Based on the questions and context provided, 
        you should think step by step, make reasonable inferences based on the context, 
        and give the most reasonable answer.
        when referencing information from the background knowledge, 
        cite the appropriate source(s) using their corresponding numbers,
        the answer should include at least one source citation,
        only cite a source when you are explicitly referencing it,
        then output the result in JSON format as the following, only choose an status from the list:[REVISED, UNCONFIDENT]:: 
        {"Status": "REVISED if you successfully inferred the correct answer, else UNCONFIDENT", "Answer": "The revised answer", "Reason": "The explanation for your reasoning process"}
        Respond only, do not explain yourself or output anything else.
        '''

    def rectify(self, question: str, samples: List):
        context = "\n".join(samples)
        useer_instruction = f"Question: {question}\n, Context: {context}"
        messages = [SystemMessage(self.instruction), HumanMessage(useer_instruction)]
        messages = ChatPromptTemplate.from_messages(messages).format_prompt()
        total_tokens = 0
        response_content = None
        try:
            chat_completion = self.client.invoke(messages.to_messages())
            response_content = chat_completion.content
            total_tokens = chat_completion.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            if response_content[len(response_content) - 1] == '.':
                response_content = response_content[:len(response_content) - 1]
            result = json.loads(response_content)
            result = EasyDict(result)
            return result, total_tokens
        except Exception as e:
            print(f'Failed to review the execution, cause: {e}, question: {question}, response: {response_content}')
        return {"Status": "UNCONFIDENT", "Answer": "I don't know"}, total_tokens

    def evaluate(self, question: str, answer: str, evidences: List[str]):
        if "I don't know" in answer:
            return {"Status": "UNCONFIDENT", "Score": 0, "Correctness": 0, "Credibility": 0}, 0
        total_tokens = 0
        correctness_score, consumed_tokens = self.evaluate_correctness(question, answer, evidences)
        print(f'Correctness: {correctness_score}, Consumed tokens: {consumed_tokens}')
        total_tokens += consumed_tokens
        credibility_score, consumed_tokens = self.evaluate_credibility(question, answer, evidences)
        print(f'Credibility: {credibility_score}, Consumed tokens: {consumed_tokens}')
        total_tokens += consumed_tokens
        # score = self.ratio * correctness_score + (1 - self.ratio) * credibility_score
        score = weighted_geometric_mean([correctness_score, credibility_score], [self.ratio, 1 - self.ratio])
        score = round(score, 2)
        if score >= self.threshold:
            return {"Status": "PASS", "Score": score, "Correctness": correctness_score,
                    "Credibility": credibility_score}, total_tokens
        else:
            return {"Status": "UNCONFIDENT", "Score": score, "Correctness": correctness_score,
                    "Credibility": credibility_score}, total_tokens

    def evaluate_correctness(self, question, answer, evidences):
        context = ";".join(evidences)
        total_tokens = 0
        instruction = f"""
        Given the following question, answer and context, evaluate the factual correctness on a scale from 0 to 1.
        Question: {question}
        Answer: {answer}
        Context: {context}
        Directly return the correctness score.Don't explain yourself or output anything else.
        """
        messages = [HumanMessage(instruction)]
        try:
            messages = ChatPromptTemplate.from_messages(messages).format_prompt()
            chat_completion = self.client.invoke(messages.to_messages())
            response_content = chat_completion.content
            total_tokens = chat_completion.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            return float(response_content), total_tokens
        except Exception as e:
            print(e)
            return 0.0, total_tokens

    def is_correct(self, question: str, answer: str, evidences):
        evidences = [pprint.pformat(evidence) for evidence in evidences]
        result, _ = self.evaluate(question, answer, evidences)
        logger.info(f"judge result:\t{result}")
        if result['Status'] == "PASS":
            return True
        return False

    def evaluate_credibility(self, question, answer, evidences):
        context = ";".join(evidences)
        total_tokens = 0
        input = f'''As an Attribution Validator, 
        your task is to verify whether a given reference can support the given claim. 
        A claim can be either a plain sentence or a question followed by its answer. 
        Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. 
        A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, 
        while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. 
        \n\nClaim: {question} {answer}
        \n Reference:{context}'''
        response_content = self.model(input)
        output = response_content[0]['generated_text']
        if output == "Attributable":
            return 1, total_tokens
        elif output == "Contradictory":
            return 0.001, total_tokens
        elif output == "Extrapolatory":
            return 0.5, total_tokens


def weighted_geometric_mean(data, weights):
    values = []
    for x in data:
        if x <= 0:
            x = 0.001
        values.append(x)
    values = np.array(values)
    weights = np.array(weights)
    return np.average(values, weights=weights)
