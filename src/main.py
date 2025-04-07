import logging
from fastapi import FastAPI, HTTPException
from retriever import Retriever
from llm_integration import LLMIntegration
from typing import List, Dict, Any, Union
import os


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = FastAPI()
retriever = Retriever()
llm = LLMIntegration()


def format_search_results(
    documents_nested: List[Union[List[str], str]],
    metadatas_nested: List[Union[List[Dict[str, Any]], Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Takes raw ChromaDB results and formats them with metadata.
    Flattens nested lists and zips results into a structured format.
    """
    documents = documents_nested[0] if documents_nested and isinstance(documents_nested[0], list) else documents_nested
    metadatas = metadatas_nested[0] if metadatas_nested and isinstance(metadatas_nested[0], list) else metadatas_nested

    if not isinstance(documents, list) or not isinstance(metadatas, list):
        return []

    combined = []
    for doc, meta in zip(documents, metadatas):
        source = "unknown"
        chunk_index = -1

        if isinstance(meta, dict):
            source = os.path.basename(str(meta.get("source", "unknown")))
            chunk_index = meta.get("chunk_index", -1)

        combined.append({
            "text": doc,
            "source": source,
            "chunk_index": chunk_index
        })

    return combined


@app.get("/")
def read_root() -> dict:
    """
    Root endpoint to check if the FastAPI server is running.
    """
    return {"message": "FastAPI is running!"}


@app.get("/search")
def search(query: str) -> dict:
    try:
        search_results = retriever.search(query)

        formatted = format_search_results(
            documents_nested=search_results.get("documents", []),
            metadatas_nested=search_results.get("metadatas", [])
        )

        return {"response": formatted}

    except Exception as e:
        logging.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/ai-search")
def ai_search(query: str) -> dict:
    """
    Retrieve rules and enhance them with AI.
    """
    try:
        search_results = retriever.search(query)
        documents = search_results.get("documents", [])
        if documents and isinstance(documents[0], list):
            documents = documents[0]

        retrieved_context = "\n".join(documents)

        ai_response = llm.generate_response(query, retrieved_context)
        return {"response": ai_response}
    except Exception as e:
        logging.error(f"Error during AI search: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

