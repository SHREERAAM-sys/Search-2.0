from abc import ABC, abstractmethod
from typing import List

import psycopg
from colpali_search.custom_types import VectorList
from colpali_search.database import conn_params
from colpali_search.services.search_engine.distance_metrics import (
    CosineSimilarity,
    HammingDistance,
    MaxSim,
)
from loguru import logger
from pgvector.psycopg import register_vector_async
from psycopg.cursor import Row
from psycopg.rows import dict_row


def log_row(row: Row):
    logger.info(
        f"Filename: {row['filename']}, Page number: {row['page_number']}, File ID: {row['file_id']}"
    )


async def execute_postgresql_search_command(
    command: str, query_embeddings: VectorList
) -> List[Row]:
    conn = await psycopg.AsyncConnection.connect(
        **conn_params, autocommit=True, row_factory=dict_row
    )

    async with conn:
        await register_vector_async(conn)
        r = await conn.execute(command, (query_embeddings,))
        result = await r.fetchall()

        for row in result:
            log_row(row)

        return result


class SearchStrategy(ABC):
    @abstractmethod
    async def search(self, query_embeddings: VectorList, top_k: int) -> List[Row]:
        pass


class ExactMaxSimSearchStrategy(SearchStrategy):
    async def search(self, query_embeddings: VectorList, top_k: int) -> List[Row]:
        max_sim = MaxSim()
        max_sim.calculate()

        SQL_RETRIEVE_TOP_K_DOCS = f"""
        WITH top_pages AS (
            SELECT
                e.page_id, max_sim(e.vector_embedding, %s) AS max_sim
            FROM
                embedding e
            ORDER BY
                max_sim DESC
            LIMIT {top_k}
        )
        SELECT
            p.page_number,
            p.file_id,
            f.*
        FROM
            top_pages tp
        JOIN
            page p ON p.id = tp.page_id
        JOIN
            file f ON f.id = p.file_id
        ORDER BY
            tp.max_sim DESC;
        """
        return await execute_postgresql_search_command(
            SQL_RETRIEVE_TOP_K_DOCS, query_embeddings
        )


class ANNHNSWHammingSearchStrategy(SearchStrategy):
    """Scale MaxSim using Binary Quantization and Hamming Distance

    Convert the floating point 128-dimensional vectors to 128-bit vectors
    """

    async def search(self, query_embeddings: VectorList, top_k: int) -> List[Row]:
        hamming = HammingDistance()
        hamming.calculate()

        # Rerank using cosine distance
        SQL_RERANK = f"""
        WITH hamming_results AS (
            SELECT *
            FROM (
                SELECT *
                FROM hamming(%s)
            )
            ORDER BY vector_embedding <=> query
            LIMIT {top_k}
        )
        SELECT
            p.id AS page_id,
            p.*,
            f.id AS file_id,
            f.*
        FROM
            hamming_results hr
        JOIN
            embedding e ON e.id = hr.embedding_id
        JOIN
            page p ON p.id = e.page_id
        JOIN
            file f ON f.id = p.file_id
        """
        return await execute_postgresql_search_command(SQL_RERANK, query_embeddings)


class ANNHNSWCosineSimilaritySearchStrategy(SearchStrategy):
    async def search(self, query_embeddings: VectorList, top_k: int) -> List[Row]:
        cosine_similarity = CosineSimilarity()
        cosine_similarity.calculate()

        SQL_COSINE_SIMILARITY = f"""
        WITH top_pages AS (
            SELECT
                e.page_id, cosine_similarity(e.vector_embedding, %s) AS cosine_similarity
            FROM
                embedding e
            ORDER BY
                cosine_similarity DESC
            LIMIT {top_k}
        )
        SELECT
            p.page_number,
            p.file_id,
            f.*
        FROM
            top_pages tp
        JOIN
            page p ON p.id = tp.page_id
        JOIN
            file f ON f.id = p.file_id
        ORDER BY
            tp.cosine_similarity DESC;
        """
        return await execute_postgresql_search_command(
            SQL_COSINE_SIMILARITY, query_embeddings
        )
