
import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


def process_docs(data_dir: str, db_dir: str, db_col: str):
    
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    semantic_splitter = SemanticChunker(
        embeddings, breakpoint_threshold_type="interquartile"
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
            print("File being processed: ", file.name, file.read_text())
            documents.extend(
                recursive_splitter.create_documents(file.read_text())
            )
    print("Chunking done")
    Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        path=db_dir,
        collection_name=db_col,
    )
            


data_dir = os.getenv("DATA_DIR") or "./docs"
db_dir = os.getenv("QDRANT_DIR") or "./db_docs"
db_col = os.getenv("QDRANT_COL") or "extract-rag.default"
process_docs(data_dir, db_dir, db_col)