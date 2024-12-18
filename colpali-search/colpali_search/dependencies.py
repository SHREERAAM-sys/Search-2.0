from typing import Annotated

from colpali_search.context import app_context
from colpali_search.services.benchmark_service import BenchmarkService
from colpali_search.services.embedding_service import EmbeddingSerivce
from colpali_search.services.file_service import FileService
from colpali_search.services.indexing_service import IndexingService
from colpali_search.services.model_service import ColPaliModelService
from colpali_search.services.pdf_conversion_service import PDFConversionService
from colpali_search.services.search_service import SearchService
from fastapi import Depends

ModelServiceDep = Annotated[
    ColPaliModelService,
    Depends(lambda: app_context.model_service),
]

PDFConversionServiceDep = Annotated[
    PDFConversionService,
    Depends(lambda: app_context.pdf_conversion_service),
]

EmbeddingSerivceDep = Annotated[
    EmbeddingSerivce,
    Depends(lambda: app_context.embedding_service),
]

SearchSerivceDep = Annotated[
    SearchService,
    Depends(lambda: app_context.search_service),
]

FileServiceDep = Annotated[
    FileService,
    Depends(lambda: app_context.file_service),
]

IndexingServiceDep = Annotated[
    IndexingService,
    Depends(lambda: app_context.indexing_service),
]

BenchmarkServiceDep = Annotated[
    BenchmarkService,
    Depends(lambda: app_context.benchmark_service),
]
