from langchain.tools import BaseTool
from typing import List
from src.indexer.vector_store import VectorStore

class RetrievalTool(BaseTool):
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, vector_store: VectorStore, index_name: str,):
        desc = """
        用于检索文档的工具
        参数：
        query: 检索参数
        k: 最终返回的文档数量，默认为4
        """
        super().__init__(name="retrieval", description=desc)
        object.__setattr__(self, "vector_store", vector_store)
        object.__setattr__(self, "index_name", index_name)
        
    def _run(self, query: str, k: int) -> List[str]:
        return self.vector_store.search(self.index_name, query, k)
        
    async def _arun(self, query: str, k:int) -> List[str]:
        return self.vector_store.search(self.index_name, query, k)
