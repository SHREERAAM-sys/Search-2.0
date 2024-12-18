from functools import lru_cache
from typing import List, Optional

from colpali_search.config import settings
from colpali_search.services.search_service import SearchService
from datasets import load_dataset
from sqlalchemy.ext.asyncio import AsyncSession


class BenchmarkService:
    def __init__(self, search_service: SearchService):
        dataset_size = 100
        self.pages = self.fetch_dataset_column(
            column_name="page", dataset_size=dataset_size
        )
        self.queries = self.fetch_dataset_column(
            column_name="query", dataset_size=dataset_size
        )
        self.actual_pages = [int(page_number) for page_number in self.pages]
        self.search_service = search_service

    @lru_cache(maxsize=3)
    def fetch_dataset(self, dataset_size: Optional[int] = None):
        return load_dataset(
            settings.benchmark_dataset_name,
            split=f"test[:{dataset_size}]" if dataset_size else "test",
        )

    def fetch_dataset_column(
        self,
        column_name: str,
        dataset_size: Optional[int],
    ) -> List[int]:
        ds = self.fetch_dataset(dataset_size)
        return ds[column_name]

    def recall(self, actual: List[int], predicted: List[int], top_k: int) -> float:
        """Measure proportion of relevant documents that are retrieved"""
        act_set = set(actual)
        pred_set = set(predicted[:top_k])
        result = round(len(act_set & pred_set) / float(len(act_set)), 2)
        return result

    async def average_recall(
        self, top_k: int, user_id: int, session: AsyncSession
    ) -> float:
        """Mean of recall scores across multiple queries"""
        total_recall = 0
        for query, page in zip(self.queries, self.actual_pages):
            response = await self.search_service.search(
                query=query, top_k=top_k, user_id=user_id, session=session
            )
            predicted = [res["page_number"] for res in response]
            recall_score = self.recall([page], predicted, top_k)
            total_recall += recall_score
        average_recall = total_recall / len(self.queries)
        return average_recall

    async def precision(self, top_k: int, user_id: int, session: AsyncSession):
        """Measure the proportion of retrieved documents that are relevant"""
        correct = 0
        for query, page in zip(self.queries, self.actual_pages):
            response = await self.search_service.search(
                query=query, top_k=top_k, user_id=user_id, session=session
            )
            predicted = [res["page_number"] for res in response]
            if page in predicted:
                correct += 1
        return correct / len(self.queries)

    async def mrr(self, top_k: int, user_id: int, session: AsyncSession):
        """Average the reciprocal ranks of the first relevant document retrieved across multiple queries"""
        total_rank = 0
        for query, page in zip(self.queries, self.actual_pages):
            response = await self.search_service.search(
                query=query, top_k=top_k, user_id=user_id, session=session
            )
            predicted = [res["page_number"] for res in response]
            rank = 1 / (predicted.index(page) + 1) if page in predicted else 0
            total_rank += rank
        return total_rank / len(self.queries)
