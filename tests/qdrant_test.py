
import os
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, PayloadSchemaType, MatchValue
from langchain_qdrant import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from gliner2 import GLiNER2

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=4)

client = QdrantClient(url="http://localhost:6333")
collection = os.getenv("QDRANT_COL") or "extract-rag.default"

client.create_payload_index(
    collection_name=collection,
    field_name="metadata.filter",
    field_schema=PayloadSchemaType.KEYWORD
)

qdrant = Qdrant(
        embeddings=embedding_model,
        client=client,
        collection_name=collection,
    )
query = "What were the organisations with the biggest earnings in 1987 and which ones were helped by the US Government?"

start_time = time.perf_counter()

retriever = qdrant.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 6
    }
)
result1 =  retriever.invoke(query)
end_time = time.perf_counter()

print("results without filter", result1, "time elapsed: ", end_time - start_time)


extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")


filter_val = extractor.classify_text(query,
    {
        "aspects": {
            "labels": [
                "Macroeconomics", 
                "Government-Work",
                "Currencies",
                "Energy",
                "Commodities", 
                "Agriculture",
                "Livestock",
                "Corporate-Finance"
            ],
            "multi_label": True,
            "cls_threshold": 0.5
        }
    }
)["aspects"]

start_time_fil = time.perf_counter()
filtered_retriever = qdrant.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 6,
        "filter": Filter(
        must=[
            FieldCondition(key="metadata.filter", match=MatchValue(value=v))
            for v in filter_val
        ]
    )
    }
)

result2 =  filtered_retriever.invoke(query)
end_time_fil = time.perf_counter()


_match = True

if len(result1) != len(result2):
    _match = False
else:
    for i in range(len(result1)):
        if result1[i].metadata["_id"] != result2[i].metadata["_id"]:
            _match = False
            break


print("filter", filter_val)
print("results with filter", result2, "time elapsed: ", end_time_fil - start_time_fil)
print("_match", _match)