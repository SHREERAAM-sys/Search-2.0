from typing import AsyncGenerator

import pytest
import pytest_asyncio
from colpali_search.app import app, get_current_user
from colpali_search.config import settings
from colpali_search.context import initialize_context
from colpali_search.database import DatabaseConfig, get_session
from colpali_search.models import Base, User
from colpali_search.utils import get_now
from pgvector.psycopg import register_vector_async
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

test_engine = create_async_engine(
    settings.test_database_url,
    isolation_level="AUTOCOMMIT",
)


@event.listens_for(test_engine.sync_engine, "connect")
def connect(dbapi_connection, connection_record):
    dbapi_connection.run_async(register_vector_async)


test_session_maker = async_sessionmaker(bind=test_engine, expire_on_commit=False)


async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
    async with test_session_maker() as session:
        yield session


async def override_get_current_user():
    async with test_session_maker() as session:
        user = User(
            email="colpalisearch@gmail.com",
            password_hash="123",
            created_at=get_now(),
        )

        session.add(user)
        await session.commit()
        await session.refresh(user)

        return user.id


app.dependency_overrides[get_session] = override_get_session
app.dependency_overrides[get_current_user] = override_get_current_user


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def use_test_database():
    DatabaseConfig.set_dbname("colpalisearch_test")
    yield
    # Reset after tests
    DatabaseConfig.set_dbname("colpalisearch")


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    """Set up fresh database for each test"""

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    initialize_context()

    async with test_session_maker() as session:
        async with session.begin():
            yield session
            await session.rollback()
