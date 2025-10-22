"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Qdrant.
"""

import os
from contextlib import contextmanager
from typing import Generator, List

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from shared.configuration import BaseConfiguration

## Encoder constructors


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "fastembed":
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

            return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=4)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_qdrant_retriever(
    configuration: BaseConfiguration,
    embedding_model: Embeddings,
    filters: List[str] | None
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific qdrant store"""
    from langchain_qdrant import Qdrant
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    vector_store = Qdrant.from_existing_collection(
        embedding=embedding_model,
        collection_name=os.environ["QDRANT_COL"],
        url="http://localhost:6333",
    )

    if filters:
        yield vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
    else:
        yield vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
                "filter": Filter(
                    must=[
                        FieldCondition(key="metadata.filter", match=MatchValue(value=v))
                        for v in filters
                    ]
                )
            }
        )

@contextmanager
def make_retriever(
    filters: List[str] | None, 
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "qdrant":
            with make_qdrant_retriever(configuration, embedding_model, filters) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
