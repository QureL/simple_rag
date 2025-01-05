from .base_agent import BaseAgent

class FundAgent(BaseAgent):
    def get_system_message(self) -> str:
        return """你是一个基金投资助手。当用户询问基金相关问题时，你需要:
1. 使用retrieval工具搜索相关基金文档
2. 分析检索结果是否足够回答问题
3. 如果不够，继续检索或更换关键词检索,或者扩展搜索范围继续检索
4. 基于检索到的内容给出准确的答案
""" 