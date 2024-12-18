from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from searchagent.colpali.repositories.all import (
    EmbeddingRepository,
    FileRepository,
    FlattenedEmbeddingRepository,
    FolderRepository,
    PageRepository,
    QueryRepository,
)
from searchagent.colpali.schemas import ImageMetadata
from searchagent.colpali.search_engine.context import IndexingContext
from searchagent.colpali.search_engine.strategy_factory import IndexingStrategyFactory
from searchagent.db_connection import execute_analyze
from searchagent.models import Embedding, File, FlattenedEmbedding, Folder, Page, Query
from searchagent.utils import VectorList
from sqlalchemy.ext.asyncio import AsyncSession


class EmbeddingSerivce:
    def __init__(
        self,
        user_id: int,
        session: AsyncSession,
        input_dir: Optional[Path],
        benchmark_embed: bool,
    ):
        self.user_id = user_id
        self.session = session
        self.input_dir = input_dir
        self.benchmark_embed = benchmark_embed

        self.folder_repository = FolderRepository(Folder, session=session)
        self.file_repository = FileRepository(File, session=session)
        self.page_repository = PageRepository(Page, session=session)
        self.embedding_repository = EmbeddingRepository(Embedding, session=session)
        self.flattened_embedding_repository = FlattenedEmbeddingRepository(
            FlattenedEmbedding, session=session
        )
        self.query_repository = QueryRepository(Query, session=session)

    async def upsert_doc_embeddings(
        self, embeddings: List[torch.Tensor], metadata: List[ImageMetadata]
    ):
        logger.info("Initiating the upsert process for document embeddings...")
        chunk_size = 500
        embed_len = len(embeddings)

        async with self.session.begin():
            for i in range(0, embed_len, chunk_size):
                embeddings_chunk = embeddings[i : i + chunk_size]
                metadata_chunk = metadata[i : i + chunk_size]

                await self._process_chunk(embeddings_chunk, metadata_chunk)

        await self._finalize_operations()
        logger.info(
            "Successfully completed the upsert process for all document embeddings."
        )

    async def upsert_query_embeddings(
        self, query: str, query_embeddings: List[VectorList]
    ):
        await self.query_repository.add(query, query_embeddings, self.user_id)

    async def _get_or_add_folder(self) -> Folder:
        folder_name, folder_path = "", ""

        if not self.benchmark_embed:
            # Benchmark dataset (vidore/syntheticDocQA_artificial_intelligence_test)
            # doesn't have folder_name and folder_path
            folder_name = str(self.input_dir.name)
            folder_path = str(self.input_dir)

        logger.info(f"Getting or adding folder: {folder_name}")

        folder = await self.folder_repository.get_by_folder_path(folder_path)

        if not folder:
            folder = await self.folder_repository.add(
                folder_name, folder_path, self.user_id
            )
        return folder

    async def _get_or_add_file(self, folder: Folder, metadata: ImageMetadata) -> File:
        filepath, filename = metadata.filepath, metadata.filename

        logger.info(f"Getting or adding file: {filename}")

        file = await self.file_repository.get_by_filepath_filename(filepath, filename)

        if not file:
            file = await self.file_repository.add(
                filepath, filename, metadata.total_pages, folder
            )
        return file

    async def _get_or_add_page(self, file: File, metadata: ImageMetadata) -> Page:
        page_id = metadata.page_id

        page = await self.page_repository.get_by_page_id_and_file(page_id, file)

        if not page:
            page = await self.page_repository.add(page_id, file)

        return page

    async def _process_chunk(
        self, embeddings_chunk: List[torch.Tensor], metadata_chunk: List[ImageMetadata]
    ):
        for embedding, metadata in zip(embeddings_chunk, metadata_chunk):
            await self._process_single_embedding(embedding, metadata)

    async def _process_single_embedding(
        self, embedding: torch.Tensor, metadata: ImageMetadata
    ):
        embeddings = embedding.tolist()
        vector_embedding = [np.array(e) for e in embeddings]

        folder = await self._get_or_add_folder()
        file = await self._get_or_add_file(folder, metadata)
        page = await self._get_or_add_page(file, metadata)

        embedding = await self.embedding_repository.add_or_replace(
            vector_embedding, page
        )

        await self.flattened_embedding_repository.add_or_replace(
            vector_embedding, embedding
        )

    async def _finalize_operations(self):
        """Update database statistics and execute indexing"""
        await execute_analyze()

        ctx = IndexingContext(
            IndexingStrategyFactory.create_strategy("HNSWCosineSimilarity")
        )
        await ctx.execute_indexing_strategy()
