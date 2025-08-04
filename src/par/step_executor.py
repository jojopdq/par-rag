from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_util import LangChainModel


class StepExecutor:
    """
    Act Module.
    """

    def __init__(self, client: LangChainModel):
        self.client = client
        self.instruction = '''
        You are an expert at inference and citation. 
        You will read each source information carefully and cite evidence related to the current issue to answer the question.
        When referencing information from a source, cite the appropriate source(s) using their corresponding numbers. 
        Every answer should include at least one source citation. 
        Only cite a source when you are explicitly referencing it.
        If you don't know the answer, just return "I don't know".
        Directly respond your answer, do not explain your answer or distract from irrelevant information in the source, 
        and do not output anything that is not relevant to the question.
        # Examples begin
        Source 1:
        The sky is red in the evening and blue in the morning.
        Source 2:
        Water is wet when the sky is red.
        Source 3:
        Wolves and dogs belong to the species, Canis lupus.
        Query: When is water wet?
        Answer: In the evening. Water is wet when the sky is red[2], which occurs in the evening [1].
        # Examples end
        '''

    def answer(self, question: str, samples: List):
        context = "\n".join(samples)
        user_instruction = f'''
        Now it's your turn. Below are several numbered sources of information:
        \n---Source Information---\n
        {context}
        \n------\n
        Query: {question}\n
        Answer:
        '''
        messages = [SystemMessage(self.instruction), HumanMessage(user_instruction)]
        messages = ChatPromptTemplate.from_messages(messages).format_prompt()

        try:
            chat_completion = self.client.invoke(messages.to_messages())
            response_content = chat_completion.content
            total_tokens = chat_completion.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            if "So the answer is: " in response_content:
                response_content = response_content.split("So the answer is:")[1].strip()
            return response_content, total_tokens
        except Exception as e:
            print(f'Failed to answer current question:{question}, exception:{e}')
            return None

    def refine_next_question(self, next_question: str, trajectories):
        instruction = '''
        You are good at analysis and inference.
        Please only supplement the question with relevant and logical information based on the previous reasoning trajectories provided, 
        without deviating from the original question intent.
        Respond with the refined question only, do not explain yourself or output anything else.
        '''
        user_instruction = f'''
        Trajectories: {trajectories} 
        Question:{next_question}]\n 
        '''
        messages = [SystemMessage(instruction), HumanMessage(user_instruction)]
        messages = ChatPromptTemplate.from_messages(messages).format_prompt()

        try:
            chat_completion = self.client.invoke(messages.to_messages())
            response_content = chat_completion.content
            total_tokens = chat_completion.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            return response_content, total_tokens
        except Exception as e:
            print(f'Failed to answer current question:{next_question}, exception:{e}')
            return None
