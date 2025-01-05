import yaml
from typing import Dict, List
from langchain_openai import ChatOpenAI
import asyncio
from src.agents.legal_agent import LegalAgent
from src.agents.fund_agent import FundAgent
from src.agents.history_agent import HistoryAgent
from src.tools.retrieval_tool import RetrievalTool
from src.indexer.vector_store import VectorStore
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate


"""Prompt for the router chain in the multi-prompt chain."""

MULTI_PROMPT_ROUTER_TEMPLATE = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. You may also revise the original input if you \
think that revising it will ultimately lead to a better response from the language \
model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use or "DEFAULT"
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{input}

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""


class RAGController:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.vector_store = VectorStore(self.config)
        self.setup_indexes()
        self.setup_agents()
        self.setup_router()
    
    def setup_router(self):
        destinations = {
            "legal": "法律相关问题，包括合同、诉讼、法规等",
            "fund": "处理基金和投资相关问题，包括理财产品、市场分析等",
            "history": "历史问题的检索",
        }
        router_prompt = PromptTemplate(
            template=MULTI_PROMPT_ROUTER_TEMPLATE,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
            partial_variables={"destinations": str(destinations),}
        )
        self.llm = ChatOpenAI(
            model=self.config["model"]["name"],
            temperature=0
        )

        router_chain = LLMRouterChain.from_llm(
            llm=self.llm,
            prompt=router_prompt
        )

        self.router_chain = router_chain
        
    def setup_indexes(self):
        for name, store_config in self.config["vector_stores"].items():
            self.vector_store.create_index(name, store_config["path"])
            
    def setup_agents(self):
        self.agents = {
            "legal": LegalAgent(
                self.config,
                [RetrievalTool(vector_store=self.vector_store, index_name="legal")]
            ),
            "fund": FundAgent(
                self.config,
                [RetrievalTool(vector_store=self.vector_store, index_name="fund")]
            ),
            "history": HistoryAgent(
                self.config,
                [RetrievalTool(vector_store=self.vector_store, index_name="history")]
            )
        }
        
        
    async def process_query(self, query: str) -> str:
        response = self.router_chain.route({"input": query})
        destination = response.destination
        agent = self.agents.get(destination)
        if destination is None or agent is None:
            # 直接回答
            msg = self.llm.invoke(query)
            return msg.content

        return await agent.run(response.next_inputs)
        
    async def batch_process(self, queries: List[str]) -> List[str]:
        # 批量
        tasks = [self.process_query(query) for query in queries]
        return await asyncio.gather(*tasks)