import asyncio

from colpali_search.services.embedding_service import EmbeddingSerivce
from colpali_search.services.model_service import ColPaliModelService
from colpali_search.services.search_engine.context import SearchContext
from colpali_search.services.search_engine.strategy_factory import SearchStrategyFactory
from sqlalchemy.ext.asyncio import AsyncSession


class SearchService:
    def __init__(
        self, embedding_service: EmbeddingSerivce, model_service: ColPaliModelService
    ):
        self.embedding_service = embedding_service
        self.model_service = model_service

    async def search(self, user_id: int, query: str, top_k: int, session: AsyncSession):
        # Run similarity search over the page embeddings for all the pages in the collection
        # top_indices has the shape of
        # tensor([[12,  0, 14],
        # [15, 14, 11]])

        query_embeddings = await asyncio.to_thread(
            self.model_service.embed_query, query
        )
        await self.embedding_service.upsert_query_embeddings(
            user_id, query, query_embeddings, session
        )

        ctx = SearchContext(
            SearchStrategyFactory.create_strategy("ANNHNSWCosineSimilarity")
        )
        return await ctx.execute_search_strategy(query_embeddings, top_k)
