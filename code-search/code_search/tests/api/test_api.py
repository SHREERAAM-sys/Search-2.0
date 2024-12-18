import pytest
from fastapi.testclient import TestClient

from code_search.database.model import CodeFile
from code_search.main import app
from code_search.repository.file_repository import FileRepository
from code_search.tests.test_config import get_db  # Import get_db from your config file
from sqlalchemy.orm import Session

# FastAPI TestClient
client = TestClient(app)

@pytest.fixture(scope="function")
def create_sample_file():
    # Creating a mock file for testing
    file_content = "print('Hello, World!')"
    file_name = "example.py"
    return file_name, file_content

# Override the get_db dependency to use the one from the config file
app.dependency_overrides[get_db] = get_db

def test_upload_file(create_sample_file):
    file_name, file_content = create_sample_file

    # Prepare a file-like object
    file_data = {"file": (file_name, file_content, "text/plain")}

    # Make the API call to upload the file
    response = client.post("/api/v1/upload/file", files=file_data)

    # Assert the response status code
    assert response.status_code == 200
    assert response.json() == {"message": f"File '{file_name}' uploaded and indexed successfully.", "file_id": 1}

    # Optionally: Check if the file was saved in the database
    db = next(get_db())  # Using the get_db method from the test config
    file_record = db.query(FileRepository).filter(FileRepository.filename == file_name).first()
    assert file_record is not None
    assert file_record.filename == file_name
    assert file_record.content == file_content

@pytest.fixture(scope="function")
def create_sample_search_data(db: Session):
    # Create sample code files and embeddings for search test
    file1 = CodeFile(filename="file1.py", content="def add(a, b): return a + b")
    file2 = CodeFile(filename="file2.py", content="def subtract(a, b): return a - b")
    db.add(file1)
    db.add(file2)
    db.commit()
    db.refresh(file1)
    db.refresh(file2)

    # Mock the embeddings for the code files
    embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    embedding2 = [0.6, 0.7, 0.8, 0.9, 1.0]
    FileRepository.save_embedding(db, file1.id, embedding1)
    FileRepository.save_embedding(db, file2.id, embedding2)

    return file1, file2


def test_search_code(create_sample_search_data):
    # Arrange
    file1, file2 = create_sample_search_data
    query = "add function"

    # Act: Send the query to the search endpoint
    response = client.post("/api/v1/search", data={"query": query})

    # Assert
    assert response.status_code == 200
    assert "query" in response.json()
    assert response.json()["query"] == query
    assert "response" in response.json()

    # Ensure the response contains expected context (e.g., file content for similar files)
    assert "def add(a, b): return a + b" in response.json()["response"]  # Check if relevant content is in the response
