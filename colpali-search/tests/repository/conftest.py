import numpy as np
import pytest_asyncio
from colpali_search.config import settings
from colpali_search.models import (
    Base,
    Embedding,
    File,
    FlattenedEmbedding,
    IndexingStrategy,
    Page,
    Query,
    User,
)
from colpali_search.utils import get_now
from pgvector.psycopg import register_vector_async
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


@pytest_asyncio.fixture(scope="function")
async def async_session():
    async_engine = create_async_engine(
        settings.test_database_url,
        isolation_level="AUTOCOMMIT",
    )

    @event.listens_for(async_engine.sync_engine, "connect")
    def connect(dbapi_connection, connection_record):
        dbapi_connection.run_async(register_vector_async)

    async_session_maker = async_sessionmaker(bind=async_engine, expire_on_commit=False)

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()


@pytest_asyncio.fixture(scope="function")
async def seed(async_session):
    async with async_session.begin():
        VEC_DIM = 128
        vectors = [
            np.array([1.0] * VEC_DIM, dtype=np.float16),
            np.array([2.0] * VEC_DIM, dtype=np.float16),
        ]

        user = User(
            email="test@gmail.com",
            password_hash="123",
            created_at=get_now(),
        )
        async_session.add(user)

        file = File(
            filename="test.pdf",
            filetype="pdf",
            total_pages=1,
            last_modified=get_now(),
            created_at=get_now(),
        )
        async_session.add(file)

        page = Page(
            page_number=1,
            last_modified=get_now(),
            created_at=get_now(),
            file=file,
        )
        async_session.add(page)

        embedding = Embedding(
            vector_embedding=vectors,
            last_modified=get_now(),
            created_at=get_now(),
            page=page,
        )
        async_session.add(embedding)

        flattened_embedding = FlattenedEmbedding(
            vector_embedding=vectors[0],
            last_modified=get_now(),
            created_at=get_now(),
            embedding=embedding,
        )
        async_session.add(flattened_embedding)

        query = Query(
            text="test query",
            vector_embedding=vectors,
            created_at=get_now(),
            user=user,
        )
        async_session.add(query)

        indexing_strategy = IndexingStrategy(
            strategy_name="ExactMaxSim",
            created_at=get_now(),
        )
        async_session.add(indexing_strategy)

        await async_session.commit()
