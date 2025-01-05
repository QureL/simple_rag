from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from typing import List, Dict

class BaseAgent:
    def __init__(self, config: Dict, tools: List):
        self.llm = ChatOpenAI(
            model=config["model"]["name"],
            temperature=config["model"]["temperature"]
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=self.get_system_message(),
            extra_prompt_messages=[
                MessagesPlaceholder(variable_name="chat_history")
            ]
        )
        
        self.agent = OpenAIFunctionsAgent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt
        )
        
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            memory=self.memory,
            verbose=True
        )
        
    def get_system_message(self) -> str:
        raise NotImplementedError
        
    async def run(self, query: str):
        return await self.executor.arun(query) 