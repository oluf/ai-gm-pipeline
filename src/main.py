import logging
from fastapi import FastAPI, HTTPException
from src.retriever import Retriever
from src.llm_integration import LLMIntegration

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()
retriever = Retriever()
llm = LLMIntegration()

@app.get("/")
def read_root() -> dict:
    """
    Root endpoint to check if the FastAPI server is running.
    """
    return {"message": "FastAPI is running!"}

@app.get("/search")
def search(query: str) -> dict:
    """
    Retrieve relevant RPG rules based on the query.
    """
    try:
        search_results = retriever.search(query)
        logging.info(f"Query: {query}")
        logging.info(f"Search Results: {search_results}")
        return {"response": search_results}
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
        retrieved_context = "\n".join(search_results['documents'])
        ai_response = llm.generate_response(query, retrieved_context)
        return {"response": ai_response}
    except Exception as e:
        logging.error(f"Error during AI search: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")