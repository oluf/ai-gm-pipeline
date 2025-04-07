import os
import json
import hashlib
import logging
from typing import Dict
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pdfplumber

from config import CHROMADB_PATH, DB_COLLECTION, EMBEDDING_MODEL_NAME, HASH_FILE_PATH, PDF_STORE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup the ChromaDB client and collection
chromadb_client = chromadb.PersistentClient(path=CHROMADB_PATH)
collection = chromadb_client.get_or_create_collection(DB_COLLECTION)

# Download/setup the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

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
    Extract text and tables from a PDF file.
    """
    text = ""
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            tables = page.extract_tables() or []

            table_text = ""
            for table in tables:
                formatted = "\n".join([
                    " | ".join(cell or "" for cell in row)
                    for row in table if any((cell or "").strip() for cell in row)
                ])
                table_text += f"\n[Page {i+1} Table]\n{formatted}\n"

            text += page_text + table_text + f"\n(Page {i + 1})\n"
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=768,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ".",
            " ",
            ""
        ]
    )

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
