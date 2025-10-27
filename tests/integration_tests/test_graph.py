import os
from contextlib import contextmanager
from typing import Generator

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langsmith import expect, unit, _expect

from retrieval_graph import graph
from shared.configuration import BaseConfiguration
from shared.retrieval import make_text_encoder
from dotenv import load_dotenv

load_dotenv()

@contextmanager
def make_qdrant_vectorstore(
    configuration: BaseConfiguration,
) -> Generator[VectorStore, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from qdrant_client import QdrantClient
    from langchain_qdrant import Qdrant

    embedding_model = make_text_encoder(configuration.embedding_model)

    client = QdrantClient(os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection = os.getenv("QDRANT_COL", "extract-rag.default")
    
    qdrant = Qdrant(
        embeddings=embedding_model,
        client=client,
        collection_name=collection,
    )
    yield qdrant


@pytest.mark.asyncio
@unit
async def test_retrieval_graph() -> None:
    config = RunnableConfig(
        configurable={
            "retriever_provider": "qdrant",
            "embedding_model": "fastembed/BAAI/bge-base-en-v1.5",
        }
    )

    # test News-related query
    res = await graph.ainvoke(
        {"messages": [("user", "Give me some info on Wheat production in china in 1986-87? ")]},
        config,
    )
    response = str(res["messages"][-1].content)
    expect(response.lower()).to_contain("china")
    expect( response.lower().find("I am not able to find")).to_equal(-1)

