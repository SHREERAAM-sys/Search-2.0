from typing import List

from colpali_search.database import get_session
from colpali_search.dependencies import FileServiceDep
from colpali_search.schemas.endpoints.file import FileResultResponse
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

files_router = APIRouter(prefix="/files", tags=["files"])


@files_router.get("/")
async def get_all_files(
    file_service: FileServiceDep,
    session: AsyncSession = Depends(get_session),
) -> List[FileResultResponse]:
    """
    Retrieve all files stored in the database

    Args:
        file_service (FileServiceDep): Service for interacting with file data
        session (AsyncSession, optional): Database session for queries. Defaults to Depends(get_session)

    Raises:
        HTTPException: If an error occurs during retrieval

    Returns:
        List[FileResultResponse]: A list of file metadata and details
    """
    try:
        files = await file_service.get_all_files(session)
        return [FileResultResponse.model_validate(file) for file in files]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while retrieving files: {str(e)}",
        )


@files_router.get("/{id}")
async def get_file_by_id(
    id: int, file_service: FileServiceDep, session: AsyncSession = Depends(get_session)
) -> FileResultResponse:
    """
    Retrieve a file's metadata and details by its ID

    Args:
        id (int): The unique identifier of the file
        file_service (FileServiceDep): Service for interacting with file data
        session (AsyncSession, optional): Database session for queries. Defaults to Depends(get_session)

    Raises:
        HTTPException: If the file with the given ID is not found or an error occurs

    Returns:
        FileResultResponse: Metadata and details of the requested file
    """
    try:
        file = await file_service.get_file_by_id(id, session)
        return FileResultResponse.model_validate(file)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while retrieving the file with ID {id}: {str(e)}",
        )


@files_router.delete("/{id}")
async def delete_file_by_id(
    id: int,
    file_service: FileServiceDep,
    session: AsyncSession = Depends(get_session),
):
    """
    Delete a file and its associated metadata by its ID.

    Args:
        id (int): The unique identifier of the file to be deleted.
        file_service (FileServiceDep): Service for handling file operations.
        session (AsyncSession, optional): Database session for executing queries. Defaults to Depends(get_session).

    Raises:
        HTTPException: If the file with the given ID does not exist or an error occurs during deletion.

    Returns:
        dict: A message indicating successful deletion.
    """
    try:
        await file_service.delete_file_by_id(id, session)
        return {"success": "Deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while deleting the file with ID {id}: {str(e)}",
        )
