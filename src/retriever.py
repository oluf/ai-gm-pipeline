import logging
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Union, Any

from config import CHROMADB_PATH, DB_COLLECTION, EMBEDDING_MODEL_NAME, DEFAULT_N_RESULTS


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Retriever:
    def __init__(self) -> None:
        try:
            self.client = chromadb.PersistentClient(path=CHROMADB_PATH)
            self.collection = self.client.get_or_create_collection(DB_COLLECTION)
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logging.info("Retriever initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Retriever: {e}")
            raise

    def search(self, query: str, n_results: int = DEFAULT_N_RESULTS) -> Dict[str, List[List[Union[str, Dict[str, Any]]]]]:
        """
        Search for relevant RPG rules based on the given query.

        Args:
            query (str): The search query.
            n_results (int): The number of results to return.

        Returns:
            Dict[str, List[str]]: The search results containing the most relevant RPG rules.
        """
        try:
            query_embedding = self.embedding_model.encode(query)
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas"]
            )

            return {
                "documents": search_results["documents"],
                "metadatas": search_results["metadatas"]
            }
        except Exception as e:
            logging.error(f"Error during search: {e}")
            return {"documents": []}

if __name__ == "__main__":
    retriever = Retriever()
    test_query = "How do critical hits work?"
    results = retriever.search(test_query)
    logging.info(f"Search results: {results}")