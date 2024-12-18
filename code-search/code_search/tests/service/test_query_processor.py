import pytest
import numpy as np
from sqlalchemy.orm import Session
from code_search.services.search_processor import QuerySearchProcessor
from code_search.repository.file_repository import FileRepository
from code_search.database.model import CodeFile
from code_search.tests.test_config import get_db  # Importing the session config


@pytest.fixture(scope="module")
def db():
    # Use the session dependency from the test config to get the database session
    db_session = next(get_db())
    yield db_session
    db_session.close()


@pytest.fixture
def create_sample_code_file(db: Session):
    # Create a sample CodeFile and associated embedding for testing
    file = CodeFile(filename="example.py", content="print('Hello, World!')")
    db.add(file)
    db.commit()
    db.refresh(file)

    # Save a mock embedding for the file
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    FileRepository.save_embedding(db, code_file_id=file.id, embedding=embedding)

    return file


def test_fetch_embedding(db: Session, create_sample_code_file):
    # Arrange
    code_file_id = create_sample_code_file.id
    processor = QuerySearchProcessor()  # Using the processor to fetch embeddings

    # Act
    embedding = processor.fetch_embedding(db, code_file_id)

    # Assert
    assert isinstance(embedding, np.ndarray)  # Ensure it's a numpy array
    assert embedding.shape[0] == 5  # Check that the embedding has 5 elements
    assert np.array_equal(embedding, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))  # Ensure the values match the expected
