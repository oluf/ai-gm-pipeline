import os

# Define constants for Llama.cpp
LLAMA_BINARY = "/home/oluf/llama.cpp/build/bin/llama-run"
MODEL_FILE = "/home/oluf/projects/ai-gm-pipeline/models/mythomax-13B.Q4_K_M.gguf"

GPU_LAYERS = "35"
TEMPERATURE = "0.7"
TIMEOUT = 30

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMADB_PATH = os.path.join(BASE_DIR, "../data/rpg_sources_db")
HASH_FILE_PATH = os.path.join(BASE_DIR, "../data/processed_files.json")
PDF_STORE = os.path.join(BASE_DIR, "../data/pdfs")
DB_COLLECTION = "rpg_sources"

CHUNK_SIZE = 768
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = 'all-MPNET-base-v2'
DEFAULT_N_RESULTS = 3