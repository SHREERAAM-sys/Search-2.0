from abc import ABC, abstractmethod

import psycopg
from colpali_search.database import conn_params
from colpali_search.models import VECT_DIM
from pgvector.psycopg import register_vector


def execute_postgresql_distance_metrics_command(command: str):

    with psycopg.connect(**conn_params, autocommit=True) as conn:
        register_vector(conn)
        conn.execute(command)


class DistanceMetric(ABC):
    @abstractmethod
    def calculate(self):
        """Calculate similarity scores"""
        pass


class HammingDistance(DistanceMetric):
    def calculate(self):
        SQL_HAMMING_FUNC = f"""
        CREATE OR REPLACE FUNCTION hamming(query halfvec[])
        RETURNS TABLE (
            embedding_id integer,
            query halfvec({VECT_DIM}),
            vector_embedding halfvec({VECT_DIM}),
            hamming_dist integer
        ) AS $$
            WITH queries AS (
                SELECT unnest(query) AS query
            )
            SELECT
                fe.embedding_id,
                q.query,
                fe.vector_embedding,
                binary_quantize(fe.vector_embedding)::bit({VECT_DIM}) <~>
                binary_quantize(q.query) AS hamming_dist
            FROM
                queries q
            CROSS JOIN
                flattened_embedding fe
            ORDER BY
                hamming_dist ASC
            LIMIT 20
        $$ LANGUAGE SQL
        """
        execute_postgresql_distance_metrics_command(SQL_HAMMING_FUNC)


class CosineSimilarity(DistanceMetric):
    def calculate(self):
        SQL_COSINE_SIMILARITY = """
        CREATE OR REPLACE FUNCTION cosine_similarity(document halfvec[], query halfvec[])
        RETURNS double precision AS $$
            WITH queries AS (
                SELECT row_number() OVER () AS query_number, *
                FROM (SELECT unnest(query) AS query)
            ),
            documents AS (
                SELECT unnest(document) AS document
            ),
            similarities AS (
                SELECT query_number, 1 - (document <=> query) AS similarity
                FROM queries CROSS JOIN documents
            )
            SELECT SUM(similarity)
            FROM similarities
        $$ LANGUAGE SQL
        """
        execute_postgresql_distance_metrics_command(SQL_COSINE_SIMILARITY)


class MaxSim(DistanceMetric):
    def calculate(self):
        """Based on
        https://github.com/pgvector/pgvector-python/blob/master/examples/colbert/exact.py
        """

        SQL_MAXSIM_FUNC = """
        CREATE OR REPLACE FUNCTION max_sim(document halfvec[], query halfvec[])
        RETURNS double precision AS $$
            WITH queries AS (
                SELECT row_number() OVER () AS query_number, *
                FROM (SELECT unnest(query) AS query)
            ),
            documents AS (
                SELECT unnest(document) AS document
            ),
            similarities AS (
                SELECT query_number, 1 - (document <=> query) AS similarity
                FROM queries CROSS JOIN documents
            ),
            max_similarities AS (
                SELECT MAX(similarity) AS max_similarity
                FROM similarities GROUP BY query_number
            )
            SELECT SUM(max_similarity)
            FROM max_similarities
        $$ LANGUAGE SQL
        """
        execute_postgresql_distance_metrics_command(SQL_MAXSIM_FUNC)
