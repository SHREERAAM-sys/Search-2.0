from typing import List

from colpali_search.models import (
    Embedding,
    File,
    FlattenedEmbedding,
    IndexingStrategy,
    Page,
    Query,
)
from colpali_search.repository.base_repository import Repository
from colpali_search.custom_types import IndexingStrategyType, VectorList
from colpali_search.utils import get_now
from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession


class FileRepository(Repository[File]):
    async def get_by_filename(self, filename: str, session: AsyncSession) -> File:
        file_stmt = select(File).filter_by(filename=filename)
        return await session.scalar(file_stmt)

    async def get_by_id(self, id: int, session: AsyncSession) -> File:
        return await session.get(File, id)

    async def add(self, filename: str, total_pages: int, session: AsyncSession) -> File:
        file = File(
            filename=filename,
            filetype="pdf",
            total_pages=total_pages,
            last_modified=get_now(),
            created_at=get_now(),
        )
        session.add(file)
        return file


class PageRepository(Repository[Page]):
    async def get_by_page_number_and_file(
        self, page_number: int, file: File, session: AsyncSession
    ) -> Page:
        file_stmt = select(Page).filter_by(id=page_number, file_id=file.id)
        return await session.scalar(file_stmt)

    async def add(self, page_number: int, file: File, session: AsyncSession) -> Page:
        page = Page(
            page_number=page_number,
            last_modified=get_now(),
            created_at=get_now(),
            file=file,
        )
        session.add(page)
        return page


class EmbeddingRepository(Repository[Embedding]):
    async def get_by_page(self, page: Page, session: AsyncSession) -> Embedding:
        embedding_stmt = select(Embedding).filter_by(page_id=page.id)
        return await session.scalar(embedding_stmt)

    async def add(
        self, vector_embedding: VectorList, page: Page, session: AsyncSession
    ) -> Embedding:
        embedding = Embedding(
            vector_embedding=vector_embedding,
            page=page,
            last_modified=get_now(),
            created_at=get_now(),
        )
        session.add(embedding)
        return embedding

    async def delete_by_page(self, page: Page, session: AsyncSession):
        delete_stmt = delete(Embedding).where(Embedding.page_id == page.id)
        await session.execute(delete_stmt)

    async def add_or_replace(
        self, vector_embedding: VectorList, page: Page, session: AsyncSession
    ) -> Embedding:
        logger.info(f"Adding or replacing embeddings for file: {page.file.filename}")
        existing_embedding = await self.get_by_page(page, session)

        if existing_embedding:
            await self.delete_by_page(page, session)

        return await self.add(vector_embedding, page, session)


class FlattenedEmbeddingRepository(Repository[FlattenedEmbedding]):
    async def add(
        self, vector_embedding: VectorList, embedding: Embedding, session: AsyncSession
    ) -> List[FlattenedEmbedding]:
        flattened_embedding = [
            FlattenedEmbedding(
                vector_embedding=ve,
                last_modified=get_now(),
                created_at=get_now(),
                embedding=embedding,
            )
            for ve in vector_embedding
        ]
        session.add_all(flattened_embedding)
        return flattened_embedding

    async def delete_by_embedding(self, embedding: Embedding, session: AsyncSession):
        delete_stmt = delete(FlattenedEmbedding).where(
            FlattenedEmbedding.embedding_id == embedding.id
        )
        await session.execute(delete_stmt)

    async def add_or_replace(
        self, vector_embedding: VectorList, embedding: Embedding, session: AsyncSession
    ) -> List[FlattenedEmbedding]:
        await self.delete_by_embedding(embedding, session)
        return await self.add(vector_embedding, embedding, session)


class QueryRepository(Repository[Query]):
    async def add(
        self,
        query: Query,
        query_embeddings: VectorList,
        user_id: int,
        session: AsyncSession,
    ) -> Query:
        query = Query(
            text=query,
            vector_embedding=query_embeddings,
            created_at=get_now(),
            user_id=user_id,
        )
        session.add(query)
        return query


class IndexingStrategyRepository(Repository[IndexingStrategy]):
    async def add(
        self, strategy: IndexingStrategy, session: AsyncSession
    ) -> IndexingStrategy:
        session.add(strategy)
        await session.commit()
        return strategy

    async def get_current_strategy(self, session: AsyncSession) -> IndexingStrategy:
        indexing_stmt = select(IndexingStrategy).filter_by(id=1)
        return await session.scalar(indexing_stmt)

    async def configure_strategy(
        self, strategy_name: IndexingStrategyType, session: AsyncSession
    ) -> IndexingStrategy:
        name = strategy_name.alias
        strategy = await self.get_current_strategy(session)

        if strategy:
            strategy.strategy_name = name
            strategy.created_at = get_now()
            await session.commit()
        else:
            strategy = IndexingStrategy(id=1, strategy_name=name, created_at=get_now())
            await self.add(strategy, session)

        return strategy

    async def reset_strategy(self, session: AsyncSession):
        """Reset to default HNSWCosineSimilarity"""
        await self.configure_strategy(
            IndexingStrategyType.hnsw_cosine_similarity, session
        )
