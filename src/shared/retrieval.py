"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Qdrant and MongoDB.
"""

import os
from contextlib import contextmanager
from typing import Generator

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

            return FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_qdrant_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific qdrant store"""
    from langchain_qdrant import Qdrant

    vector_store = Qdrant.from_existing_collection(
        embedding=embedding_model,
        collection_name='extrag-rag.default',
        path=os.environ["QDRANT_DIR"],
    )

    yield vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )


@contextmanager
def make_mongodb_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="extrag-rag.default",
        embedding=embedding_model,
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "qdrant":
            with make_qdrant_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
