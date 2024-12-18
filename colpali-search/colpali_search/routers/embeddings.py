import asyncio
from typing import Callable, List, Optional

from colpali_search.database import get_session
from colpali_search.dependencies import (
    EmbeddingSerivceDep,
    ModelServiceDep,
    PDFConversionServiceDep,
)
from colpali_search.schemas.endpoints.embeddings import EmbeddingsResponse
from colpali_search.schemas.internal.pdf import PDFsConversion
from colpali_search.utils import convert_tensors_to_list_of_lists
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

embeddings_router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class FileValidator:
    allowed_types: List[str] = ["application/pdf"]

    @classmethod
    def _validate_file_type(cls, file: UploadFile) -> None:
        file_type = file.content_type
        if file_type not in cls.allowed_types:
            raise HTTPException(
                status_code=404,
                detail=f"Invalid file type. Expected: {cls.allowed_types}. "
                f"Received: '{file_type}'.",
            )

    @classmethod
    def validate_file(cls, file: UploadFile) -> UploadFile:
        if not file:
            raise HTTPException(
                status_code=422,
                detail="File not provided. Please upload a valid file.",
            )
        cls._validate_file_type(file)
        return file

    @classmethod
    def validate_files(cls, files: List[UploadFile]) -> List[UploadFile]:
        for file in files:
            cls._validate_file_type(file)
        return files


@embeddings_router.post("/file")
async def generate_embeddings_for_file(
    model_service: ModelServiceDep,
    embedding_service: EmbeddingSerivceDep,
    pdf_conversion_service: PDFConversionServiceDep,
    file: UploadFile = Depends(FileValidator.validate_file),
    session: AsyncSession = Depends(get_session),
) -> EmbeddingsResponse:
    """Generates embeddings for an uploaded PDF file

    Args:
        model_service (ModelServiceDep): Handles model operations for embedding generation
        embedding_service (EmbeddingSerivceDep): Manages embedding creation and storage
        pdf_conversion_service (PDFConversionServiceDep): Converts PDF to images for embedding
        file (UploadFile): Validated PDF file. Defaults to Depends(FileValidator.validate_file)
        session (AsyncSession): Database session for embedding operations. Defaults to Depends(get_session)

    Raises:
        HTTPException: On processing, embedding, or database errors

    Returns:
        EmbeddingsResponse: Details about the generated embeddings
    """
    try:
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None, pdf_conversion_service.convert_single_pdf2image, file
        )

        images, metadata = result.single_pdf_images, result.metadata
        embeddings = model_service.embed_images([images], [metadata])

        await embedding_service.upsert_doc_embeddings(
            embeddings=embeddings, metadata=metadata, session=session
        )

        return EmbeddingsResponse(
            message="File successfully embedded",
            embeddings=convert_tensors_to_list_of_lists(embeddings),
            metadata=metadata,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error during embedding: {e}"
        )


@embeddings_router.post("/files")
async def generate_embeddings_for_files(
    model_service: ModelServiceDep,
    embedding_service: EmbeddingSerivceDep,
    pdf_conversion_service: PDFConversionServiceDep,
    files: List[UploadFile] = Depends(FileValidator.validate_files),
    session: AsyncSession = Depends(get_session),
) -> EmbeddingsResponse:
    """Generates embeddings for a list of uploaded PDF files

    Args:
        model_service (ModelServiceDep): Handles model operations for embedding generation
        embedding_service (EmbeddingSerivceDep): Manages embedding creation and storage
        pdf_conversion_service (PDFConversionServiceDep): Converts PDFs to images for embedding
        files (List[UploadFile]): Validated PDF files. Defaults to Depends(FileValidator.validate_files)
        session (AsyncSession): Database session for embedding operations. Defaults to Depends(get_session)

    Returns:
        EmbeddingsResponse: Details about the embeddings generated for the files.
    """

    def pdf_conversion_func():
        return pdf_conversion_service.convert_pdfs2image(files)

    return await process_embeddings(
        model_service, embedding_service, pdf_conversion_func, session
    )


@embeddings_router.post("/benchmark")
async def generate_embeddings_for_benchmark(
    model_service: ModelServiceDep,
    embedding_service: EmbeddingSerivceDep,
    pdf_conversion_service: PDFConversionServiceDep,
    session: AsyncSession = Depends(get_session),
) -> EmbeddingsResponse:
    """Generate embeddings for benchmark datasets

    Args:
        model_service (ModelServiceDep): Provides the model for generating embeddings
        embedding_service (EmbeddingSerivceDep):
            Handles embedding creation and database storage
        pdf_conversion_service (PDFConversionServiceDep):
            Converts benchmark datasets (e.g., PDFs) to images for embedding
        session (AsyncSession, optional): Database session for storing embeddings.
            Defaults to Depends(get_session)

    Returns:
        EmbeddingsResponse: Summary of generated embeddings for benchmark datasets
    """
    pdf_conversion_func = pdf_conversion_service.retrieve_pdfImage_from_vidore
    return await process_embeddings(
        model_service, embedding_service, pdf_conversion_func, session
    )


async def process_embeddings(
    model_service: ModelServiceDep,
    embedding_service: EmbeddingSerivceDep,
    pdf_conversion_func: Callable[[Optional[List[UploadFile]]], PDFsConversion],
    session: AsyncSession,
) -> EmbeddingsResponse:
    """Process embeddings for given files by converting PDFs, generating embeddings, and storing them

    Args:
        model_service (ModelServiceDep): Service for generating embeddings from images
        embedding_service (EmbeddingSerivceDep): Handles storing embeddings and associated metadata
        pdf_conversion_func (Callable[[Optional[List[UploadFile]]], PDFsConversion]):
            Function to convert PDFs to images and metadata
        session (AsyncSession): Database session for operations

    Raises:
        HTTPException: If an unexpected error occurs during processing

    Returns:
        EmbeddingsResponse: Contains success message, embeddings, and metadata
    """
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, pdf_conversion_func)

        images_list, metadata_list = result.images_list, result.metadata_list

        embeddings = model_service.embed_images(images_list, metadata_list)
        metadata = [item for sublist in metadata_list for item in sublist]

        await embedding_service.upsert_doc_embeddings(
            embeddings=embeddings, metadata=metadata, session=session
        )

        return EmbeddingsResponse(
            message="Files successfully embedded",
            embeddings=convert_tensors_to_list_of_lists(embeddings),
            metadata=metadata,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error during embedding: {e}"
        )
