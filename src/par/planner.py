import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

MAX_RETRIES = 3


class Planner(object):
    """
    Plan Module. It's responsible to create plan for complex question.
    """

    def __init__(self, client):
        self.client = client
        self.instruction = '''
        You are a thoughtful expert who is very good at breaking down complex problems by developing a plan.
        For each step in the plan, generate a sub-problem, each of which must be clear and independently executable, 
        and all of which are logically coherent with the goal of solving the complex problem.
        Do not use any external knowledge, assumptions, or information beyond what is explicitly stated in the context. 
        # Format instructions 
        Use the following Strict JSON format, only choose an action from the list:[Retrieve, Answer]: 
        [
          { 
            "Thought":"[your thought about the current step]"
            "Question":"[the question you generated for the current step]"
            "Action": "[the action you chose]"
          }
        ]  
        '''

    def create(self, query: str, history_record=None, context=None):
        """
        @param query: query str
        @return: plan from question
        """

        if history_record:
            user_prompt_with_few_shot = f'''
                    # Example Begin
                    Here is an example,
                    Question: {history_record.get("question")}
                    Generated Plan: {history_record.get("plan")}
                    # Example End
                    '''
        else:
            user_prompt_with_few_shot = ""

        user_instruction = f'''
                Think carefully, build a step-by-step plan for this [Question]:{query}.
                Don't use your intrinsic knowledge about the sub-question.
                Return only your answer, do not explain it and do not output anything that is not relevant to the answer. 
                '''

        messages = [SystemMessage(f"{self.instruction}\n{user_prompt_with_few_shot}"), HumanMessage(user_instruction)]
        messages = ChatPromptTemplate.from_messages(messages).format_prompt()
        retries = 0
        total_tokens = 0
        response_content = None
        while retries < MAX_RETRIES:
            try:
                chat_completion = self.client.invoke(messages.to_messages())
                response_content = chat_completion.content
                if response_content.startswith("```json"):
                    response_content = response_content.replace("```json", "")
                    response_content = response_content.replace("```", "")
                steps = json.loads(response_content)
                idx = 1
                plan = []
                for step in steps:
                    item = {f'Step {idx}': step}
                    plan.append(item)
                    idx += 1
                total_tokens += chat_completion.response_metadata.get('token_usage', {}).get('total_tokens', 0)
                return plan, total_tokens
            except Exception as e:
                print(f'Failed to create plan, cause: {e}, query: {query}, response: {response_content}')
                retries += 1
        return None, 0
