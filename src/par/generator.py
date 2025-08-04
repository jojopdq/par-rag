from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate


class Generator:
    def __init__(self, client):
        self.client = client
        self.instruction = '''
        You serve as an intelligent assistant, 
        adept at facilitating users through complex, 
        multi-hop reasoning across multiple trajectories, 
        each consisting of a document set including a reasoning thought, answer and relevant information. 
        Your task is to generate the last answer for the question by referring to the trajectories.
        If you don't know the answer, just return "I don't know".
        Else, respond the final answer only, do not explain yourself or output anything else.
        
        # Example start
        Question: What is the distance in kilometers between Tokyo and Osaka, rounded to the nearest whole number? 
        So the answer is: 550 
        # Example end
        '''

    def read(self, question: str, trajectories):
        user_instruction = f"Trajectories: \n{trajectories}\nQuestion: \n{question}\nSo the answer is:"
        messages = [SystemMessage(self.instruction), HumanMessage(user_instruction)]
        messages = ChatPromptTemplate.from_messages(messages).format_prompt()

        chat_completion = self.client.invoke(messages.to_messages())
        response_content = chat_completion.content
        total_tokens = chat_completion.response_metadata.get('token_usage', {}).get('total_tokens', 0)
        if "So the answer is: " in response_content:
            response_content = response_content.split("So the answer is:")[1].strip()
        return response_content, total_tokens
