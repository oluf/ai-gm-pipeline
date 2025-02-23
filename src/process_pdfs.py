import os
import json
import hashlib
from typing import IO

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pdfplumber



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_CHROMADB_PATH = os.path.join(BASE_DIR, "../data/rpg_sources_db")
HASH_FILE_PATH = os.path.join(BASE_DIR, "../data/processed_files.json")
PDF_STORE = os.path.join(BASE_DIR, "../data/pdfs")
DB_COLLECTION = "rpg_sources"

""" Setup the ChromaDB client and collection """
chromadb_path = os.path.join(BASE_DIR, PROJECT_CHROMADB_PATH)
chromadb_client = chromadb.PersistentClient(path=chromadb_path)
collection = chromadb_client.get_or_create_collection(DB_COLLECTION)


""" Download/setup the embedding model """
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')



def confirm_project_paths():
    """
       Confirm the existence of required project paths and files.

       This function checks for the existence of:
        1. The JSON hash file
        2. The PDF file store directory

       If they do not exist, it creates them. This ensures that the necessary paths and files are in place
       before processing any PDFs.

       Returns:
           None
    """

    print("\n\n\n1. Checking project paths...")
    print("----------------------------\n")

    """ check for the json hash file """
    print ("Checking: ", HASH_FILE_PATH)
    if not os.path.exists(HASH_FILE_PATH):
        print(f"{HASH_FILE_PATH} does not existing. Creating now.\n")
        with open(HASH_FILE_PATH, "w") as f:
            f.write("{}")
    else:
        print(f"{HASH_FILE_PATH} exists. Skipping.\n")

    """ Check for the PDF file store """
    print("Checking: ", PDF_STORE)
    if not os.path.exists(PDF_STORE):
        print(f"{PDF_STORE} does not existing. Creating now.\n")
        os.makedirs(PDF_STORE)
    else:
        print(f"{PDF_STORE} exists. Skipping.\n")



def load_processed_files():
    """
        Load the dictionary of processed file hashes from a JSON file.

        This function reads the JSON file specified by `HASH_FILE_PATH` and returns
        a dictionary where the keys are file paths and the values are their corresponding
        hash values. If the file does not exist, an empty dictionary is returned.

        Returns:
            dict: A dictionary containing file paths and their hash values.
    """

    print("2. Load list of processed files\n")
    with open(HASH_FILE_PATH, "r") as f:
        return json.load(f)



def compute_file_hash(file):
    """
        Compute the MD5 hash of a file.

        This function reads the contents of the specified file and computes its MD5 hash.
        The hash can be used to track changes to the file.

        Args:
            file (str): The path to the file to be hashed.

        Returns:
            str: The MD5 hash of the file.
    """

    hasher = hashlib.md5()

    with open(file, "rb") as f:
        hasher.update(f.read())

    return hasher.hexdigest()



def extract_text_from_pdf(file):
    """
        Extract text from a PDF file.

        This function uses pdfplumber to read the contents of the specified PDF file
        and extract the text from each page. The extracted text is concatenated into
        a single string, with page numbers appended for reference.

        Args:
            file (str): The path to the PDF file to be processed.

        Returns:
            str: The extracted text from the PDF file.
    """

    text = ""
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            text += page.extract_text() + f"\n(Page{i + 1})\n"
    return text.strip()



def save_processed_files(processed_files_data):
    """Save the updated dictionary of processed file hashes."""
    with open(HASH_FILE_PATH, "w") as f:
        json.dump(processed_files_data, f) # type: IO[str]



def chunk_and_store_pdf(file):
    """
        Split text from a PDF file into chunks, generate embeddings, and store them in ChromaDB.

        This function extracts text from the specified PDF file, splits the text into smaller chunks,
        generates embeddings for each chunk using a pre-trained embedding model, and stores the chunks
        along with their embeddings and metadata in a ChromaDB collection.

        Args:
            file (str): The path to the PDF file to be processed.

        Returns:
            None
    """

    #   Generate chunks from extracted text
    text = extract_text_from_pdf(file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(text)

    #   Generate embeddings for the text chunks
    embeddings = embedding_model.encode(chunks)

    #   Store the chunks and embeddings (and metadata) in ChromaDB
    ids = []
    metadata = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{os.path.basename(file)}_chunk_{i}"
        ids.append(chunk_id)
        metadata.append({"source": file, "chunk_index":i})

    collection.add(ids=ids, documents=chunks, embeddings=embeddings,metadatas=metadata)
    print(f"Stored {len(chunks)} chunks from {file} in ChromaDB\n")


if __name__ == "__main__":

    """ 1. Check required project paths and files to ensure that everything is in place before processing """
    confirm_project_paths()

    """ 2. Load the list of processed files """
    processed_files = load_processed_files()
    print("Processed files: ", processed_files)

    """ 3. Check and process PDFs """
    for file_name in os.listdir(PDF_STORE):
        file_path = os.path.join(PDF_STORE, file_name)

        if not file_name.lower().endswith(".pdf"):
            continue

        file_hash = compute_file_hash(file_path)

        if processed_files.get(file_path) == file_hash:
            print(f"No changes detected in {file_name}. Skipping.\n")

        else:
            print(f"Processing new or updated file: {file_name}\n")
            chunk_and_store_pdf(file_path)
            processed_files[file_path] = file_hash

    save_processed_files(processed_files)

