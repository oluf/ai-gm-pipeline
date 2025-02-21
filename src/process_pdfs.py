import os
import json
import hashlib


import chromadb
from sentence_transformers import SentenceTransformer


# import pdfplumber


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_CHROMADB_PATH = os.path.join(BASE_DIR, "../data/rpg_sources_db")
HASH_FILE_PATH = os.path.join(BASE_DIR, "../data/processed_files.json")
PDF_STORE = os.path.join(BASE_DIR, "../data/pdfs")
DB_COLLECTION = "rpg_sources"

"""
Setup Chroma DB:
    1. Get the path to the database (or where the database will be)
    2. Create a new database client that will be used for the life of the script
    3. Get the collection if it exists, otherwise create a new collection (in this case called 'rpg_sources')
"""
chromadb_path = os.path.join(BASE_DIR, PROJECT_CHROMADB_PATH)
chromadb_client = chromadb.PersistentClient(path=chromadb_path)
collection = chromadb_client.get_or_create_collection(DB_COLLECTION)


"""
Setup the embedding model that will be used to generate embeddings (vector representations) of text chunks
Note: the first time you run this script, it will download the embedding model if it doesn't already exist
"""
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


"""
Confirm required files and folders exist before starting processing
"""
def confirm_project_paths():
    print("1. Checking project paths...")
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



"""
Generate a dictionary of files that have already been processed
"""
def load_processed_files():
    print("2. Load list of processed files\n")
    with open(HASH_FILE_PATH, "r") as f:
        return json.load(f)



def compute_file_hash(file):
    hasher = hashlib.md5()

    with open(file, "rb") as f:
        hasher.update(f.read())

    return hasher.hexdigest()



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
        print(file_hash)







