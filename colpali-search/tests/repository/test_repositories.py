import numpy as np
import pytest
from colpali_search.models import (
    Embedding,
    File,
    FlattenedEmbedding,
    IndexingStrategy,
    Page,
    Query,
)
from colpali_search.repository.repositories import (
    EmbeddingRepository,
    FileRepository,
    FlattenedEmbeddingRepository,
    IndexingStrategyRepository,
    PageRepository,
    QueryRepository,
)
from colpali_search.custom_types import IndexingStrategyType
from colpali_search.utils import get_now
from sqlalchemy import select


@pytest.mark.asyncio
async def test_file_repository_get_by_filename(seed, async_session):
    filename = "test.pdf"
    file_repo = FileRepository(File)
    file = await file_repo.get_by_filename(filename, async_session)

    assert file.filename == filename


@pytest.mark.asyncio
async def test_file_repository_get_by_id(seed, async_session):
    file_repo = FileRepository(File)
    file = await file_repo.get_by_id(id=1, session=async_session)

    assert file.filename == "test.pdf"


@pytest.mark.asyncio
async def test_file_repository_add(seed, async_session):
    file_repo = FileRepository(File)
    filename, total_pages = "hnsw.pdf", 10
    file = await file_repo.add(filename, total_pages, async_session)

    assert file.filename == filename
    assert file.total_pages == total_pages


@pytest.mark.asyncio
async def test_page_repository_get_by_page_number_and_file(seed, async_session):
    page_repo = PageRepository(Page)
    file_repo = FileRepository(File)
    file = await file_repo.get_by_id(id=1, session=async_session)
    page = await page_repo.get_by_page_number_and_file(1, file, async_session)

    assert page.file.id == file.id


@pytest.mark.asyncio
async def test_page_repository_add(seed, async_session):
    page_repo = PageRepository(Page)
    file_repo = FileRepository(File)
    file = await file_repo.get_by_id(id=1, session=async_session)
    page = await page_repo.add(2, file, session=async_session)

    assert page.file.filename == file.filename


@pytest.mark.asyncio
async def test_embedding_repository_get_by_page(seed, async_session):
    embedding_repo = EmbeddingRepository(Embedding)
    page_repo = PageRepository(Page)
    file_repo = FileRepository(File)
    file = await file_repo.get_by_id(id=1, session=async_session)
    page = await page_repo.get_by_page_number_and_file(1, file, async_session)
    embedding = await embedding_repo.get_by_page(page, async_session)

    assert np.array_equal(
        embedding.vector_embedding[0].tolist(), np.array([1.0] * 128, dtype=np.float16)
    )


@pytest.mark.asyncio
async def test_delete_by_page(seed, async_session):
    # Create repositories
    page_repo = PageRepository(Page)
    embedding_repo = EmbeddingRepository(Embedding)

    # Add a file
    file = await async_session.execute(select(File))
    file = file.scalar()

    # Add two pages
    page1 = await page_repo.add(1, file, async_session)
    page2 = await page_repo.add(2, file, async_session)

    # Add embeddings for each page
    embedding1 = Embedding(
        page=page1,
        vector_embedding=[np.array([1.0] * 128, dtype=np.float32)],
        last_modified=get_now(),
        created_at=get_now(),
    )
    embedding2 = Embedding(
        page=page2,
        vector_embedding=[np.array([2.0] * 128, dtype=np.float32)],
        last_modified=get_now(),
        created_at=get_now(),
    )
    async_session.add_all([embedding1, embedding2])
    await async_session.commit()

    # Verify embeddings exist before deletion
    result = await async_session.execute(select(Embedding))
    embeddings_before = result.scalars().all()
    assert len(embeddings_before) == 3  # All three embeddings should exist

    # Delete embeddings for page1
    await embedding_repo.delete_by_page(page1, async_session)
    await async_session.commit()

    # Verify only embedding2 and embedding from conftest remain
    result = await async_session.execute(select(Embedding))
    embeddings_after = result.scalars().all()
    assert len(embeddings_after) == 2
    assert embeddings_after[1].page_id == page2.id  # Verify it's the correct one


@pytest.mark.asyncio
async def test_add_or_replace_no_existing_embedding(seed, async_session):

    embedding_repo = EmbeddingRepository(Embedding)
    page_repo = PageRepository(Page)

    file = await async_session.execute(select(File))
    file = file.scalar()
    page = await page_repo.add(1, file, async_session)

    vector_embedding = [np.array([1.0] * 128, dtype=np.float32)]
    new_embedding = await embedding_repo.add_or_replace(
        vector_embedding, page, async_session
    )

    result = await async_session.execute(
        select(Embedding).where(Embedding.page_id == page.id)
    )
    embeddings = result.scalars().all()

    assert len(embeddings) == 1
    assert embeddings[0].id == new_embedding.id
    assert np.array_equal(embeddings[0].vector_embedding[0], vector_embedding[0])


@pytest.mark.asyncio
async def test_add_or_replace_existing_embedding(seed, async_session):
    embedding_repo = EmbeddingRepository(Embedding)
    page_repo = PageRepository(Page)

    file = await async_session.execute(select(File))
    file = file.scalar()
    page = await page_repo.add(1, file, async_session)

    old_vector_embedding = [np.array([1.0] * 128, dtype=np.float32)]
    old_embedding = await embedding_repo.add(old_vector_embedding, page, async_session)

    new_vector_embedding = [np.array([2.0] * 128, dtype=np.float32)]
    new_embedding = await embedding_repo.add_or_replace(
        new_vector_embedding, page, async_session
    )

    result = await async_session.execute(
        select(Embedding).where(Embedding.page_id == page.id)
    )
    embeddings = result.scalars().all()

    assert len(embeddings) == 2
    assert embeddings[1].id == new_embedding.id
    assert np.array_equal(embeddings[1].vector_embedding[0], new_vector_embedding[0])

    assert old_embedding.id != new_embedding.id


@pytest.mark.asyncio
async def test_delete_by_embedding(seed, async_session):
    EmbeddingRepository(Embedding)
    flattened_embedding_repo = FlattenedEmbeddingRepository(FlattenedEmbedding)

    file = await async_session.execute(select(File))
    file = file.scalar()
    page = Page(page_number=1, file=file, last_modified=get_now(), created_at=get_now())
    async_session.add(page)
    await async_session.commit()

    embedding = Embedding(
        page=page,
        vector_embedding=[np.array([1.0] * 128, dtype=np.float32)],
        last_modified=get_now(),
        created_at=get_now(),
    )
    async_session.add(embedding)
    await async_session.commit()

    # Add multiple FlattenedEmbeddings for this embedding
    flattened1 = FlattenedEmbedding(
        embedding=embedding,
        vector_embedding=np.array([1.0] * 128, dtype=np.float32),
        last_modified=get_now(),
        created_at=get_now(),
    )
    flattened2 = FlattenedEmbedding(
        embedding=embedding,
        vector_embedding=np.array([2.0] * 128, dtype=np.float32),
        last_modified=get_now(),
        created_at=get_now(),
    )
    async_session.add_all([flattened1, flattened2])
    await async_session.commit()

    # Verify the FlattenedEmbeddings exist
    result = await async_session.execute(
        select(FlattenedEmbedding).where(
            FlattenedEmbedding.embedding_id == embedding.id
        )
    )
    flattened_embeddings = result.scalars().all()
    assert len(flattened_embeddings) == 2

    # Call delete_by_embedding
    await flattened_embedding_repo.delete_by_embedding(embedding, async_session)
    await async_session.commit()

    # Verify the FlattenedEmbeddings are deleted
    result = await async_session.execute(
        select(FlattenedEmbedding).where(
            FlattenedEmbedding.embedding_id == embedding.id
        )
    )
    flattened_embeddings_after = result.scalars().all()
    assert len(flattened_embeddings_after) == 0

    # Verify no other FlattenedEmbeddings were affected
    other_result = await async_session.execute(select(FlattenedEmbedding))
    all_flattened_embeddings = other_result.scalars().all()
    assert len(all_flattened_embeddings) == 1


@pytest.mark.asyncio
async def test_add_or_replace(seed, async_session):
    flattened_embedding_repo = FlattenedEmbeddingRepository(FlattenedEmbedding)
    EmbeddingRepository(Embedding)

    result = await async_session.execute(select(Embedding))
    embedding = result.scalar()
    assert embedding is not None

    result = await async_session.execute(
        select(FlattenedEmbedding).where(
            FlattenedEmbedding.embedding_id == embedding.id
        )
    )
    flattened_embeddings = result.scalars().all()
    assert len(flattened_embeddings) == 1
    original_flattened = flattened_embeddings[0]

    new_vector_embedding = [
        np.array([3.0] * 128, dtype=np.float16),
        np.array([4.0] * 128, dtype=np.float16),
    ]
    await flattened_embedding_repo.add_or_replace(
        vector_embedding=new_vector_embedding,
        embedding=embedding,
        session=async_session,
    )
    await async_session.commit()

    result = await async_session.execute(
        select(FlattenedEmbedding).where(
            FlattenedEmbedding.embedding_id == embedding.id
        )
    )
    flattened_embeddings_after = result.scalars().all()
    assert len(flattened_embeddings_after) == 2
    assert all(
        np.array_equal(fe.vector_embedding, ve)
        for fe, ve in zip(flattened_embeddings_after, new_vector_embedding)
    )

    assert original_flattened.id not in [fe.id for fe in flattened_embeddings_after]


@pytest.mark.asyncio
async def test_query_repository_add(seed, async_session):
    query_repo = QueryRepository(Query)

    query_text = "What is the meaning of life?"
    query_embeddings = [
        np.array([1.0] * 128, dtype=np.float16),
        np.array([2.0] * 128, dtype=np.float16),
    ]
    user_id = 1

    new_query = await query_repo.add(
        query_text, query_embeddings, user_id, async_session
    )
    await async_session.commit()

    # Verify the Query object was added
    result = await async_session.execute(select(Query).where(Query.id == new_query.id))
    added_query = result.scalar()

    assert added_query is not None
    assert added_query.text == query_text
    assert added_query.user_id == user_id
    assert len(added_query.vector_embedding) == len(query_embeddings)
    assert np.array_equal(added_query.vector_embedding[0], query_embeddings[0])
    assert np.array_equal(added_query.vector_embedding[1], query_embeddings[1])


@pytest.mark.asyncio
async def test_indexing_strategy_repository_add(seed, async_session):
    repo = IndexingStrategyRepository(IndexingStrategy)

    strategy = IndexingStrategy(strategy_name="ExactMaxSim", created_at=get_now())
    added_strategy = await repo.add(strategy, async_session)
    await async_session.commit()

    result = await async_session.execute(
        select(IndexingStrategy).where(IndexingStrategy.id == added_strategy.id)
    )
    retrieved_strategy = result.scalar()

    assert retrieved_strategy is not None
    assert retrieved_strategy.strategy_name == "ExactMaxSim"


@pytest.mark.asyncio
async def test_indexing_strategy_repository_get_current_strategy(seed, async_session):
    repo = IndexingStrategyRepository(IndexingStrategy)

    current_strategy = await repo.get_current_strategy(async_session)

    assert current_strategy is not None
    assert current_strategy.id == 1
    assert current_strategy.strategy_name == "ExactMaxSim"


@pytest.mark.asyncio
async def test_indexing_strategy_repository_configure_strategy(seed, async_session):
    repo = IndexingStrategyRepository(IndexingStrategy)

    await repo.configure_strategy(
        IndexingStrategyType.hnsw_binary_quantization_hamming_distance,
        session=async_session,
    )
    await async_session.commit()

    result = await async_session.execute(
        select(IndexingStrategy).where(IndexingStrategy.id == 1)
    )
    updated_strategy = result.scalar()

    assert updated_strategy is not None
    assert updated_strategy.strategy_name == "HNSWBQHamming"


@pytest.mark.asyncio
async def test_indexing_strategy_repository_reset_strategy(seed, async_session):
    repo = IndexingStrategyRepository(IndexingStrategy)

    await repo.reset_strategy(async_session)
    await async_session.commit()

    result = await async_session.execute(
        select(IndexingStrategy).where(IndexingStrategy.id == 1)
    )
    reset_strategy = result.scalar()

    assert reset_strategy is not None
    assert reset_strategy.strategy_name == "HNSWCosineSimilarity"
