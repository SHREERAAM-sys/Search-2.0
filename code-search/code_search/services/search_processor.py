import numpy as np
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from code_search.database.model import CodeFile
from code_search.repository.search_repository import QueryRepository
from code_search.repository.file_repository import FileRepository


class QuerySearchProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.query_repo = QueryRepository()  # Instantiate the query repository
        self.file_repo = FileRepository()  # Instantiate the file repository

    def fetch_embedding(self, db: Session, code_file_id: int) -> np.ndarray:
        """
        Fetch the embedding vector for a given code file ID.
        """
        embedding = self.file_repo.fetch_embedding(db, code_file_id)
        return np.array(embedding)

    def search_similar_files(self, db: Session, query: str, top_k: int = 5):
        """
        Search for the top_k most similar files to the given query based on embedding similarity.
        """
        query_embedding = self.model.encode(query)
        files = db.query(CodeFile).all()
        results = []

        for file in files:
            file_embedding = self.fetch_embedding(db, file.id)
            similarity = np.dot(query_embedding, file_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(file_embedding)
            )
            results.append((file.id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_context_from_files(self, db: Session, file_ids):
        """
        Retrieve and concatenate the content of files with the given IDs.
        """
        files = db.query(CodeFile).filter(CodeFile.id.in_(file_ids)).all()
        return "\n".join(file.content for file in files)

    def get_cached_search_result(self, db: Session, query: str):
        """
        Check if the query is cached in the database and return the result if found.
        """
        return self.query_repo.fetch_search_query(db, query)

    def save_search_result(self, db: Session, query: str, search_results: list, response: str):
        """
        Save the search result in the database cache.
        """
        self.query_repo.save_search_query(db, query, search_results, response)
