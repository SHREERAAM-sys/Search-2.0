from colpali_search.services.benchmark_service import BenchmarkService
from colpali_search.services.embedding_service import EmbeddingSerivce
from colpali_search.services.file_service import FileService
from colpali_search.services.indexing_service import IndexingService
from colpali_search.services.model_service import ColPaliModelService
from colpali_search.services.pdf_conversion_service import PDFConversionService
from colpali_search.services.search_service import SearchService


class AppContext:
    model_service: ColPaliModelService
    embedding_service: EmbeddingSerivce
    pdf_conversion_service: PDFConversionService
    search_service: SearchService
    file_service: FileService
    indexing_service: IndexingService
    benchmark_service: BenchmarkService


app_context = AppContext()


def initialize_context():
    app_context.model_service = ColPaliModelService()
    app_context.indexing_service = IndexingService()
    app_context.file_service = FileService()
    app_context.embedding_service = EmbeddingSerivce(
        indexing_service=app_context.indexing_service
    )
    app_context.search_service = SearchService(
        embedding_service=app_context.embedding_service,
        model_service=app_context.model_service,
    )
    app_context.benchmark_service = BenchmarkService(
        search_service=app_context.search_service
    )
    app_context.pdf_conversion_service = PDFConversionService(
        benchmark_service=app_context.benchmark_service
    )
