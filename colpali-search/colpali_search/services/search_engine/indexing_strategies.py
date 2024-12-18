from abc import ABC, abstractmethod
from typing import List

import psycopg
from colpali_search.models import VECT_DIM
from loguru import logger
from pgvector.psycopg import register_vector_async
from psycopg.cursor import Row
from colpali_search.database import conn_params

SQL_GET_INDEXES = """
SELECT indexname
FROM pg_indexes
WHERE schemaname = 'public'
AND tablename = 'flattened_embedding'
AND indexname != 'flattened_embedding_pkey'
"""


async def execute_postgresql_indexing_command(command: str):
    conn = await psycopg.AsyncConnection.connect(**conn_params, autocommit=True)
    async with conn:
        await register_vector_async(conn)
        await conn.execute(command)


class IndexingStrategy(ABC):
    @abstractmethod
    async def build_index(self):
        pass

    async def get_indexes(self) -> List[Row]:
        conn = await psycopg.AsyncConnection.connect(**conn_params, autocommit=True)

        async with conn:
            await register_vector_async(conn)
            r = await conn.execute(SQL_GET_INDEXES)
            result = await r.fetchall()

            for row in result:
                logger.info(f"Getting index: {row}")

            return result

    async def drop_indexes(self):
        SQL_DROP_INDEX = f"""
        DO $$
        DECLARE
            index_name TEXT;
        BEGIN
            FOR index_name IN
                {SQL_GET_INDEXES}
            LOOP
                EXECUTE format('DROP INDEX IF EXISTS %I', index_name);
            END LOOP;
        END $$;
        """
        await execute_postgresql_indexing_command(SQL_DROP_INDEX)


class HNSWIndexingBinaryQuantizationHammingDistance(IndexingStrategy):
    async def build_index(self):
        """Use expression indexing for binary quantization"""

        SQL_INDEXING = f"""
            CREATE INDEX ON flattened_embedding
            USING hnsw ((binary_quantize(vector_embedding)::bit({VECT_DIM})) bit_hamming_ops);
            """
        await execute_postgresql_indexing_command(SQL_INDEXING)


class HNSWIndexingCosineSimilarity(IndexingStrategy):
    async def build_index(self):
        SQL_INDEXING = f"""CREATE INDEX ON flattened_embedding
        USING hnsw ((vector_embedding::halfvec({VECT_DIM})) halfvec_cosine_ops);
        """
        await execute_postgresql_indexing_command(SQL_INDEXING)
