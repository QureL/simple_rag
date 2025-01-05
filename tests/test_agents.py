import pytest
from src.controller.controller import RAGController

@pytest.mark.asyncio
async def test_legal_query():
    controller = RAGController("config/config.yaml")
    result = await controller.process_query("年羹尧的父亲是谁")
    assert result is not None
    print(result)

@pytest.mark.asyncio
async def test_fund_query():
    controller = RAGController("config/config.yaml")
    result = await controller.process_query("国泰基金的基金托管人是？")
    assert result is not None
    print(result)

@pytest.mark.asyncio
async def test_batch_processing():
    controller = RAGController("config/config.yaml")
    queries = [
        "盛宣怀的生卒年月",
        "夫妻在婚姻关系存续期间所得的哪些财产，归夫妻共同所有？",
        "今天天气怎么样？"
    ]
    results = await controller.batch_process(queries)
    assert len(results) == len(queries)
    assert all(result is not None for result in results) 