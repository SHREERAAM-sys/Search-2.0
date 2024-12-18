import numpy as np
import pytest
from colpali_search.models import (
    Embedding,
    File,
    FlattenedEmbedding,
    IndexingStrategy,
    Page,
    Query,
    User,
)
from sqlalchemy import select


@pytest.mark.asyncio
async def test_file_orm(seed, async_session):
    file_stmt = select(File)
    page_stmt = select(Page)
    file = await async_session.scalar(file_stmt)
    page = await async_session.scalar(page_stmt)

    assert file.filename == "test.pdf"
    assert file.pages[0].id == page.id


@pytest.mark.asyncio
async def test_page_orm(seed, async_session):
    page_stmt = select(Page)
    page = await async_session.scalar(page_stmt)

    assert page.file_id == 1
    assert page.page_number == 1


@pytest.mark.asyncio
async def test_embedding_orm(seed, async_session):
    embedding_stmt = select(Embedding)
    page_stmt = select(Page)
    embedding = await async_session.scalar(embedding_stmt)
    page = await async_session.scalar(page_stmt)

    assert embedding.id == page.id
    assert np.array_equal(
        embedding.vector_embedding[0].tolist(), np.array([1.0] * 128, dtype=np.float16)
    )


@pytest.mark.asyncio
async def test_flattened_embedding_orm(seed, async_session):
    stmt = select(FlattenedEmbedding)
    flattened_embedding = await async_session.scalar(stmt)

    assert np.array_equal(
        flattened_embedding.vector_embedding.tolist(),
        np.array([1.0] * 128, dtype=np.float16),
    )


@pytest.mark.asyncio
async def test_query_orm(seed, async_session):
    stmt = select(Query)
    query = await async_session.scalar(stmt)

    assert np.array_equal(
        query.vector_embedding[1].tolist(),
        np.array([2.0] * 128, dtype=np.float16),
    )


@pytest.mark.asyncio
async def test_user_orm(seed, async_session):
    stmt = select(User)
    user = await async_session.scalar(stmt)

    assert user.email == "test@gmail.com"


@pytest.mark.asyncio
async def test_indexing_strategy(seed, async_session):
    stmt = select(IndexingStrategy)
    indexing_strategy = await async_session.scalar(stmt)

    assert indexing_strategy.strategy_name == "ExactMaxSim"
