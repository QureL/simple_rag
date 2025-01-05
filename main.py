
from src.controller.controller import RAGController
import asyncio
async def main():
    controller = RAGController("config/config.yaml")
    
    result = await controller.process_query("国泰基金的基金托管人是？")
    print(result)
    
    # 批量
    # queries = [
    #     "盛宣怀的生卒年月",
    #     "夫妻在婚姻关系存续期间所得的哪些财产，归夫妻共同所有？",
    #     "今天天气怎么样？"
    # ]
    # results = await controller.batch_process(queries)
    # for query, result in zip(queries, results):
    #     print(f"Q: {query}\nA: {result}\n")

if __name__ == "__main__":
    asyncio.run(main()) 