from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from code_search.database.model import CodeFile
from code_search.repository.file_repository import FileRepository


class FileEmbeddingProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        self.file_repo = FileRepository()  # Instantiate the file repository

    def embed_file(self, db: Session, filename: str, content: str) -> CodeFile:
        # Generate the embedding
        embedding = self.generate_embedding(content)

        # Save the file metadata
        file_record = self.file_repo.save_file_metadata(db, filename, content)

        # Save the embedding as individual rows
        self.file_repo.save_embedding(db, file_record.id, embedding)

        # Return the saved file record
        return file_record

    def generate_embedding(self, content: str) -> list[float]:
        # Generate and convert embedding to floats
        return [float(value) for value in self.model.encode(content)]
