import os
import json
import hashlib
import logging
from typing import Dict, List, IO
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pdfplumber

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_CHROMADB_PATH = os.path.join(BASE_DIR, "../data/rpg_sources_db")
HASH_FILE_PATH = os.path.join(BASE_DIR, "../data/processed_files.json")
PDF_STORE = os.path.join(BASE_DIR, "../data/pdfs")
DB_COLLECTION = "rpg_sources"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Setup the ChromaDB client and collection
chromadb_path = os.path.join(BASE_DIR, PROJECT_CHROMADB_PATH)
chromadb_client = chromadb.PersistentClient(path=chromadb_path)
collection = chromadb_client.get_or_create_collection(DB_COLLECTION)

# Download/setup the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def confirm_project_paths() -> None:
    """
    Confirm the existence of required project paths and files.
    """
    logging.info("Checking project paths...")

    if not os.path.exists(HASH_FILE_PATH):
        logging.info(f"{HASH_FILE_PATH} does not exist. Creating now.")
        with open(HASH_FILE_PATH, "w") as f:
            f.write("{}")
    else:
        logging.info(f"{HASH_FILE_PATH} exists. Skipping.")

    if not os.path.exists(PDF_STORE):
        logging.info(f"{PDF_STORE} does not exist. Creating now.")
        os.makedirs(PDF_STORE)
    else:
        logging.info(f"{PDF_STORE} exists. Skipping.")

def load_processed_files() -> Dict[str, str]:
    """
    Load the dictionary of processed file hashes from a JSON file.
    """
    logging.info("Loading list of processed files.")
    if os.path.exists(HASH_FILE_PATH):
        with open(HASH_FILE_PATH, "r") as f:
            return json.load(f)
    return {}

def compute_file_hash(file: str) -> str:
    """
    Compute the MD5 hash of a file.
    """
    hasher = hashlib.md5()
    with open(file, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def extract_text_from_pdf(file: str) -> str:
    """
    Extract text from a PDF file.
    """
    text = ""
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            text += page.extract_text() + f"\n(Page {i + 1})\n"
    return text.strip()

def save_processed_files(processed_files_data: Dict[str, str]) -> None:
    """
    Save the updated dictionary of processed file hashes to a JSON file.
    """
    with open(HASH_FILE_PATH, "w") as f:
        json.dump(processed_files_data, f)

def chunk_and_store_pdf(file: str) -> None:
    """
    Split text from a PDF file into chunks, generate embeddings, and store them in ChromaDB.
    """
    text = extract_text_from_pdf(file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)

    embeddings = embedding_model.encode(chunks)

    ids = []
    metadata = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{os.path.basename(file)}_chunk_{i}"
        ids.append(chunk_id)
        metadata.append({"source": file, "chunk_index": i})

    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadata)
    logging.info(f"Stored {len(chunks)} chunks from {file} in ChromaDB")

if __name__ == "__main__":
    confirm_project_paths()
    processed_files = load_processed_files()
    logging.info(f"Processed files: {processed_files}")

    for file_name in os.listdir(PDF_STORE):
        file_path = os.path.join(PDF_STORE, file_name)

        if not file_name.lower().endswith(".pdf"):
            continue

        file_hash = compute_file_hash(file_path)

        if processed_files.get(file_path) == file_hash:
            logging.info(f"No changes detected in {file_name}. Skipping.")
        else:
            logging.info(f"Processing new or updated file: {file_name}")
            chunk_and_store_pdf(file_path)
            processed_files[file_path] = file_hash

    save_processed_files(processed_files)