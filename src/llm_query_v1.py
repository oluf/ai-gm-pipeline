import os
import subprocess
import logging
from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer

# Define path and filename constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMADB_PATH = os.path.join(BASE_DIR, "../data/rpg_sources_db")
DB_COLLECTION = "rpg_sources"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

#   Configure the ChromaDB client and collection
client = chromadb.PersistentClient(path=CHROMADB_PATH)
collection = client.get_or_create_collection(DB_COLLECTION)

#   Load the SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define llama-cpp constants
LLAMA_BINARY = "/home/oluf/bin/llama.cpp/build/bin/llama-run"
MODEL_FILE = "/home/oluf/projects/ai-gm-pipeline/models/mythomax-13B.Q4_K_M.gguf"
GPU_LAYERS = "35"
TEMPERATURE = "0.7"


@app.get("/")
def read_root():
    """
        Root endpoint to check if FastAPI is running.

        Returns:
            dict: A message indicating that FastAPI is running.
    """
    return {"message": "FastAPI is running!"}