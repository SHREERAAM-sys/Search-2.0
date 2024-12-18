from sqlalchemy.orm import Session
from code_search.database.model import CodeFile, VectorEmbedding


class FileRepository:
    @staticmethod
    def save_file_metadata(db: Session, filename: str, content: str) -> CodeFile:
        # Create and save the file metadata
        file_record = CodeFile(filename=filename, content=content)
        db.add(file_record)
        db.commit()
        db.refresh(file_record)  # Refresh to access the auto-generated ID
        return file_record

    @staticmethod
    def save_embedding(db: Session, code_file_id: int, embedding: list[float]):
        # Save each embedding value as a row
        for i, value in enumerate(embedding):
            vector_record = VectorEmbedding(code_file_id=code_file_id, dimension_index=i, value=value)
            db.add(vector_record)
        db.commit()

    @staticmethod
    def fetch_embedding(db: Session, code_file_id: int) -> list[float]:
        vectors = db.query(VectorEmbedding).filter(
            VectorEmbedding.code_file_id == code_file_id
        ).order_by(VectorEmbedding.dimension_index).all()
        return [v.value for v in vectors]
