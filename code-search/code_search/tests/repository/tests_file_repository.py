import pytest
from sqlalchemy.orm import Session
from code_search.tests.test_config import Base, engine, SessionLocal
from code_search.repository.file_repository import FileRepository


@pytest.fixture(scope="function")
def db() -> Session:
    # Create the database schema
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

    # Drop the database schema after each test
    Base.metadata.drop_all(bind=engine)


def test_save_file_metadata(db: Session):
    # Arrange
    filename = "example.py"
    content = "print('Hello, World!')"

    # Act
    file = FileRepository.save_file_metadata(db, filename, content)

    # Assert
    assert file.id is not None
    assert file.filename == filename
    assert file.content == content


def test_save_embedding(db: Session):
    # Arrange
    file = FileRepository.save_file_metadata(db, "example.py", "print('Hello')")
    embedding = [0.1, 0.2, 0.3]

    # Act
    FileRepository.save_embedding(db, file.id, embedding)

    # Assert
    result = FileRepository.fetch_embedding(db, file.id)
    assert result == embedding
