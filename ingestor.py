
import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType, VectorParams, Distance
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


load_dotenv()

def extract_with_gliner(documents: List[Document]) -> List[Document]:
    from gliner2 import GLiNER2

    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    
    seen = set()
    all_labels = []
    no_extraction = 0
    
    all_label_texts =os.getenv("LABELS").split(",")

    print("all_labels", all_labels)
    for document in documents:
        print("Document beforeeeeeeeeeeeeee", document.page_content)
        labels = extractor.classify_text(
            document.page_content,
            {
                "aspects": {
                    "labels": all_label_texts,
                    "multi_label": True,
                    "cls_threshold": 0.5
                }
            }
        )["aspects"]
        document.metadata["filter"] = labels[:4]
        print("Document After", document)
        print("labels", labels, "\n**************************************************************************\\n")

        if len(labels) == 0:
            no_extraction += 1
        else:
            for x in labels:
                if x not in seen:
                    seen.add(x)
                    all_labels.append(x)
    
    print("all labels used", all_labels, "no_extraction", no_extraction)  
    return documents


def process_docs(data_dir: str, db_url: str, db_col: str):
    embeddings = FastEmbedEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"), threads=4)

    client = QdrantClient(url=db_url)
    
    client.create_collection(
        collection_name=db_col,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    client.create_payload_index(
        collection_name=db_col,
        field_name="metadata.filter",
        field_schema=PayloadSchemaType.KEYWORD
    )

    qdrant = Qdrant(
        embeddings=embeddings,
        client=client,
        collection_name=db_col,
    )
    
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=128,
        add_start_index=True,
    )

    p = Path(data_dir)
    documents = []
    for file in p.iterdir(): 
        if file.is_file():
            documents.extend(
                recursive_splitter.create_documents([file.read_text()])
            )
    print("Chunking done")
    
    extract_with_gliner(documents)
    print("extraction done", documents[1:5])

    qdrant.add_documents(
        documents=documents,
    )
    
    collection_info = client.get_collection(collection_name=db_col)
    print(f"Collection InFO: {collection_info}")

data_dir = os.getenv("DATA_DIR", "./docs")
db_url = os.getenv("QDRANT_URL", "http://localhost:6333")
db_col = os.getenv("QDRANT_COL", "extract-rag.default")
process_docs(data_dir, db_url, db_col)