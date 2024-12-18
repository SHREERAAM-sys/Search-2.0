# Code Search AI Tool

This is a FastAPI-based application that enables users to efficiently search, analyze, and manage code files across various supported programming languages and formats.

## Endpoints

### Root Endpoint
- **URL:** `/`
- **Method:** GET
- **Description:** Provides a welcome message and information about the supported file extensions.

### Upload File
- **URL:** `/api/v1/upload/file`
- **Method:** POST
- **Description:** Uploads a single file, processes the file content, and generates embeddings for the file.
- **Parameters:**
  - `file: UploadFile` - The file to be uploaded.

### Upload Multiple Files
- **URL:** `/api/v1/upload/files`
- **Method:** POST
- **Description:** Uploads multiple files, processes the file content, and generates embeddings for each file.
- **Parameters:**
  - `files: List[UploadFile]` - The list of files to be uploaded.

### Search Code
- **URL:** `/api/v1/search`
- **Method:** POST
- **Description:** Searches for similar code snippets based on the provided query.
- **Parameters:**
  - `query: str` - The search query.

