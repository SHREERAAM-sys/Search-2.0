from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, HTTPException, Form, Depends
from sqlalchemy.orm import Session
from code_search.database.session import SessionLocal
from code_search.services.file_processor import FileEmbeddingProcessor
from code_search.services.search_processor import QuerySearchProcessor
from code_search.services.rag import RAGProcessor

# Initialize the FastAPI app
app = FastAPI(title="Code Search AI Tool", version="1.0")

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize the processors
file_embedding_processor = FileEmbeddingProcessor()
query_processor = QuerySearchProcessor()
rag_processor = RAGProcessor()

# List of supported file extensions
SUPPORTED_EXTENSIONS = {
    ".py",  # Python
    ".java",  # Java
    ".js",  # JavaScript
    ".ts",  # TypeScript
    ".cpp", ".c", ".h",  # C/C++
    ".cs",  # C#
    ".rb",  # Ruby
    ".php",  # PHP
    ".html", ".css", ".scss", ".xml",  # Web files
    ".json", ".yaml", ".yml",  # Config files
    ".sql",  # SQL files
    ".go",  # Go
    ".rs",  # Rust
    ".kt",  # Kotlin
    ".swift",  # Swift
    ".sh", ".bash",  # Shell scripts
    ".r",  # R
    ".pl",  # Perl
    ".lua",  # Lua
    ".m", ".mat",  # MATLAB
    ".txt"  # Text files
}

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Code Search AI Tool",
        "description": (
            "This tool enables users to efficiently search, analyze, and manage code files "
            "across various supported programming languages and formats. It streamlines the "
            "code review process and ensures compatibility with multiple file extensions."
        ),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS)
    }

# Upload file and process embeddings
@app.post("/api/v1/upload/file")
async def upload_file(file: UploadFile, db: Session = Depends(get_db)):
    file_extension = file.filename.split(".")[-1]
    if f".{file_extension}" not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{file_extension}")

    # Read and decode the uploaded file content
    content = await file.read()
    content_str = content.decode("utf-8")

    # Save file content and generate embeddings
    file_record = file_embedding_processor.embed_file(db=db, filename=file.filename, content=content_str)

    return {"message": f"File '{file.filename}' uploaded and indexed successfully.", "file_id": file_record.id}


@app.post("/api/v1/upload/files")
async def upload_files(files: List[UploadFile], db: Session = Depends(get_db)):
    # Check for unsupported extensions
    unsupported_extensions = [
        f".{file.filename.split('.')[-1]}" for file in files
        if f".{file.filename.split('.')[-1]}" not in SUPPORTED_EXTENSIONS
    ]

    if unsupported_extensions:
        return {
            "message": "Some files have unsupported extensions.",
            "unsupported_extensions": list(set(unsupported_extensions)),  # Remove duplicates
        }

    # Process valid files
    results = []
    for file in files:
        # Read and decode the uploaded file content
        content = await file.read()
        content_str = content.decode("utf-8")

        # Save file content and generate embeddings
        file_record = file_embedding_processor.embed_file(db=db, filename=file.filename, content=content_str)

        results.append({"filename": file.filename, "file_id": file_record.id})

    return {"message": "Files uploaded and indexed successfully.", "files": results}


@app.post("/api/v1/search")
def search_code(query: str = Form(...), db: Session = Depends(get_db)):
    # Check if the query already exists in the cache
    # cached_result = query_processor.get_cached_search_result(db=db, query=query)

    # if cached_result:
    #     return {"query": query, "response": cached_result.response}

    # Search for similar files if not cached
    search_results = query_processor.search_similar_files(db=db, query=query)

    if not search_results:
        raise HTTPException(status_code=404, detail="No relevant code snippets found")

    # Extract file IDs and fetch context
    file_ids = [result[0] for result in search_results]
    context = query_processor.get_context_from_files(db=db, file_ids=file_ids)

    # # Generate RAG response using OpenAI
    # response = rag_processor.generate_rag_response(query=query, context=context)

    # # Cache the query, result, and response
    # query_processor.save_search_result(db=db, query=query, search_results=search_results, response=response)

    # return {"query": query, "response": response}
    return context

