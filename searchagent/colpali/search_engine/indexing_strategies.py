from abc import ABC, abstractmethod

import psycopg
from pgvector.psycopg import register_vector_async
from searchagent.db_connection import DBNAME
from searchagent.models import VECT_DIM


class IndexingStrategy(ABC):
    @abstractmethod
    async def build_index(self):
        pass


class HNSWIndexingBinaryQuantizationHammingDistance(IndexingStrategy):
    async def build_index(self):
        """Use expression indexing for binary quantization"""

        SQL_INDEXING = f"""
            CREATE INDEX ON flattened_embedding
            USING hnsw ((binary_quantize(vector_embedding)::bit({VECT_DIM})) bit_hamming_ops);
            """

        conn = await psycopg.AsyncConnection.connect(dbname=DBNAME, autocommit=True)

        async with conn:
            await register_vector_async(conn)
            await conn.execute(SQL_INDEXING)


class HNSWIndexingCosineSimilarity(IndexingStrategy):
    async def build_index(self):
        SQL_INDEXING = f"""CREATE INDEX ON flattened_embedding
        USING hnsw ((vector_embedding::halfvec({VECT_DIM})) halfvec_cosine_ops);
        """

        conn = await psycopg.AsyncConnection.connect(dbname=DBNAME, autocommit=True)

        async with conn:
            await register_vector_async(conn)
            await conn.execute(SQL_INDEXING)
