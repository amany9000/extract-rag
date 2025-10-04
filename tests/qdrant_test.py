
import os
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=4)

client = QdrantClient(url="http://localhost:6333")

qdrant = QdrantVectorStore(
        embedding=embedding_model,
        client=client,
        collection_name=os.getenv("QDRANT_COL") or "extract-rag.default",
    )
retriever = qdrant.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)

query = "What were the organisations with the biggest earnings in 1987?"

print("results", retriever.invoke(query))
