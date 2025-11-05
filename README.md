# ExtractRAG: The GLiNER-based Metadata-Filtered RAG

[![Open in - LangGraph Studio](https://img.shields.io/badge/Open_in-LangGraph_Studio-00324d.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4NS4zMzMiIGhlaWdodD0iODUuMzMzIiB2ZXJzaW9uPSIxLjAiIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHBhdGggZD0iTTEzIDcuOGMtNi4zIDMuMS03LjEgNi4zLTYuOCAyNS43LjQgMjQuNi4zIDI0LjUgMjUuOSAyNC41QzU3LjUgNTggNTggNTcuNSA1OCAzMi4zIDU4IDcuMyA1Ni43IDYgMzIgNmMtMTIuOCAwLTE2LjEuMy0xOSAxLjhtMzcuNiAxNi42YzIuOCAyLjggMy40IDQuMiAzLjQgNy42cy0uNiA0LjgtMy40IDcuNkw0Ny4yIDQzSDE2LjhsLTMuNC0zLjRjLTQuOC00LjgtNC44LTEwLjQgMC0xNS4ybDMuNC0zLjRoMzAuNHoiLz48cGF0aCBkPSJNMTguOSAyNS42Yy0xLjEgMS4zLTEgMS43LjQgMi41LjkuNiAxLjcgMS44IDEuNyAyLjcgMCAxIC43IDIuOCAxLjYgNC4xIDEuNCAxLjkgMS40IDIuNS4zIDMuMi0xIC42LS42LjkgMS40LjkgMS41IDAgMi43LS41IDIuNy0xIDAtLjYgMS4xLS44IDIuNi0uNGwyLjYuNy0xLjgtMi45Yy01LjktOS4zLTkuNC0xMi4zLTExLjUtOS44TTM5IDI2YzAgMS4xLS45IDIuNS0yIDMuMi0yLjQgMS41LTIuNiAzLjQtLjUgNC4yLjguMyAyIDEuNyAyLjUgMy4xLjYgMS41IDEuNCAyLjMgMiAyIDEuNS0uOSAxLjItMy41LS40LTMuNS0yLjEgMC0yLjgtMi44LS44LTMuMyAxLjYtLjQgMS42LS41IDAtLjYtMS4xLS4xLTEuNS0uNi0xLjItMS42LjctMS43IDMuMy0yLjEgMy41LS41LjEuNS4yIDEuNi4zIDIuMiAwIC43LjkgMS40IDEuOSAxLjYgMi4xLjQgMi4zLTIuMy4yLTMuMi0uOC0uMy0yLTEuNy0yLjUtMy4xLTEuMS0zLTMtMy4zLTMtLjUiLz48L3N2Zz4=)](https://langgraph-studio.vercel.app/templates/open?githubUrl=https://github.com/amany9000/extract-rag)

This is a starter project to help you get started with developing a GLiNER-based Metadata-Filtered RAG Research agent using [LangGraph](https://github.com/langchain-ai/langgraph) in [LangSmith Studio](https://docs.langchain.com/oss/python/langgraph/studio).

* [GLiNER](https://github.com/fastino-ai/GLiNER2) is an efficient model used for Named Entity Recognition(NER), Classification and Extraction. It has excellent support for CPU.

* Since using LLMs to filter unstructured data (Articles, Legal Docs, Reports etc) can be very costly, GLiNER-based Filtered RAG pipeline provide an efficient and robust solution.

* In `ingestor.py`, the data is first chunked into LangChain Documents, these documents are then classified using GLiNER, the classified labels are stored in the document's metadata and then finally document indexing in the VectorDB is performed.

* At the time of retrieval, the LLM sends back multiple (default 3) queries and their corresponding filters(if any), which are then used to retrieve data from the VectorDB.

![Graph view in LangGraph studio UI](./static/studio_ui.png)

## What it does

This project has two graphs:

* a "retrieval" graph (`src/retrieval_graph/graph.py`)
* a "researcher" subgraph (part of the retrieval graph) (`src/retrieval_graph/researcher_graph/graph.py`)


The retrieval graph manages a chat history and responds based on the fetched documents. Specifically, it:

1. Takes a user **query** as input
2. Then the researcher subgraph runs these steps:
- it first generates a list of queries (default 3) along with metadata filters (if any).
- it then retrieves the relevant documents in parallel for all queries+filters and return the documents to the LLM.
4. Finally, the LLM generates a response based on the retrieved documents and the conversation context.

## Getting Started
1. Create a `.env` file:

```bash
cp .env.example .env
```

2. Setup [Qdrant](https://github.com/qdrant/qdrant):
```
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant 
```
Qdrant is a fast vectordb. It has an extensive support for metadata filtering.

3. Install Dependencies:

```bash
uv sync
```

4. Ingest Documents from `./docs`:
```
uv run python ingestor.py
```
The documents in `./docs` are processed versions of the .sgm files of the [Reuters-21578](https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) text categorization data collection.

5. Start Langsmith Studio:
```
uv run langgraph dev --allow-blocking
```

6. Next, open the `retrieval_graph` using the dropdown in the top-left. Ask it questions about LangChain to confirm it can fetch the required information!

### Setup Model

The default values for `response_model`, `query_model` are shown below:

```yaml
response_model: google_genai/gemini-2.0-flash-lite
query_model: google_genai/gemini-2.0-flash-lite
```

#### Google AI Studio

To use Google Gemini's chat models:

1. Sign up for an [Google AI Studio API key](https://aistudio.google.com/).
2. Once you have your API key, add it to your `.env` file:
```
GOOGLE_API_KEY=your-api-key
```

### Setup Embedding Model

The default values for `embedding_model` are shown below:

```yaml
embedding_model: fastembed/BAAI/bge-base-en-v1.5
```


## How to customize

You can customize this retrieval agent template in several ways:

1. **Modify the embedding model**: You can change the embedding model used for document indexing and query embedding by updating the `embedding_model` in the configuration. Options include various fastembed models.

2. **Customize the response generation**: You can modify the `response_system_prompt` to change how the agent formulates its responses. This allows you to adjust the agent's personality or add specific instructions for answer generation.

3. **Change the language model**: Update the `response_model` in the configuration to use different language models for response generation. Options include various Claude models from Anthropic, as well as models from other providers like Fireworks AI.

4. **Extend the graph**: You can add new nodes or modify existing ones in the `src/retrieval_graph/graph.py` file to introduce additional processing steps or decision points in the agent's workflow.

5. **Add tools**: Implement tools to expand the researcher agent's capabilities beyond simple retrieval generation.

