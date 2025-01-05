from .base_agent import BaseAgent

class HistoryAgent(BaseAgent):
    def get_system_message(self) -> str:
        return """你是个历史通，当用户咨询你历史相关问题时，
1. 使用retrieval工具搜索相关历史
2. 分析检索结果是否足够回答问题
3. 如果不够，继续检索或更换关键词检索
4. 基于检索到的内容给出准确的答案
""" 