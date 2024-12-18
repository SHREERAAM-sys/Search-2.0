from typing import List

import numpy as np
import torch
from colpali_search.database import execute_analyze
from colpali_search.models import (
    Embedding,
    File,
    FlattenedEmbedding,
    IndexingStrategy,
    Page,
    Query,
)
from colpali_search.repository.repositories import (
    EmbeddingRepository,
    FileRepository,
    FlattenedEmbeddingRepository,
    IndexingStrategyRepository,
    PageRepository,
    QueryRepository,
)
from colpali_search.schemas.internal.pdf import ImageMetadata
from colpali_search.services.indexing_service import IndexingService
from colpali_search.custom_types import IndexingStrategyType, VectorList
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession


class EmbeddingSerivce:
    def __init__(self, indexing_service: IndexingService):
        self.indexing_service = indexing_service

        self.file_repository = FileRepository(File)
        self.page_repository = PageRepository(Page)
        self.embedding_repository = EmbeddingRepository(Embedding)
        self.flattened_embedding_repository = FlattenedEmbeddingRepository(
            FlattenedEmbedding
        )
        self.query_repository = QueryRepository(Query)
        self.indexing_strategy_repository = IndexingStrategyRepository(IndexingStrategy)

    async def upsert_doc_embeddings(
        self,
        embeddings: List[torch.Tensor],
        metadata: List[ImageMetadata],
        session: AsyncSession,
    ):
        """Process document embeddings in chunks"""
        logger.info("Initiating the upsert process for document embeddings...")
        chunk_size = 500
        embed_len = len(embeddings)

        for i in range(0, embed_len, chunk_size):
            embeddings_chunk = embeddings[i : i + chunk_size]
            metadata_chunk = metadata[i : i + chunk_size]

            await self._process_chunk(embeddings_chunk, metadata_chunk, session)

        await self._finalize_operations(session)
        logger.info(
            "Successfully completed the upsert process for all document embeddings."
        )

    async def upsert_query_embeddings(
        self,
        user_id: int,
        query: str,
        query_embeddings: List[VectorList],
        session: AsyncSession,
    ):
        await self.query_repository.add(query, query_embeddings, user_id, session)

    async def _get_or_add_file(
        self, metadata: ImageMetadata, session: AsyncSession
    ) -> File:
        filename = metadata.filename

        logger.info(f"Getting or adding file: {filename}")

        file = await self.file_repository.get_by_filename(filename, session)

        if not file:
            file = await self.file_repository.add(
                filename, metadata.total_pages, session
            )
        return file

    async def _get_or_add_page(
        self, file: File, metadata: ImageMetadata, session: AsyncSession
    ) -> Page:
        page_number = metadata.page_number

        page = await self.page_repository.get_by_page_number_and_file(
            page_number, file, session
        )

        if not page:
            page = await self.page_repository.add(page_number, file, session)

        return page

    async def _process_chunk(
        self,
        embeddings_chunk: List[torch.Tensor],
        metadata_chunk: List[ImageMetadata],
        session: AsyncSession,
    ):
        for embedding, metadata in zip(embeddings_chunk, metadata_chunk):
            await self._process_single_embedding(embedding, metadata, session)
            await session.commit()

    async def _process_single_embedding(
        self, embedding: torch.Tensor, metadata: ImageMetadata, session: AsyncSession
    ):
        embeddings = embedding.tolist()
        vector_embedding = [np.array(e) for e in embeddings]

        file = await self._get_or_add_file(metadata, session)
        page = await self._get_or_add_page(file, metadata, session)

        embedding = await self.embedding_repository.add_or_replace(
            vector_embedding, page, session
        )

        await self.flattened_embedding_repository.add_or_replace(
            vector_embedding, embedding, session
        )

    async def _finalize_operations(self, session: AsyncSession):
        """Update database statistics and execute indexing"""
        await execute_analyze()
        current_strategy = await self.indexing_strategy_repository.get_current_strategy(
            session
        )

        if current_strategy:
            strategy_type = IndexingStrategyType.from_alias(
                current_strategy.strategy_name
            )

            if strategy_type == IndexingStrategyType.exact_maxsim:
                return

            await self.indexing_service.build_index(strategy_type)
        else:
            # Use default HNSW with cosine similarity
            await self.indexing_service.build_index()
