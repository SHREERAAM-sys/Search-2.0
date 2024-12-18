from typing import AsyncGenerator
from urllib.parse import urlparse

from colpali_search.config import settings
from loguru import logger
from pgvector.psycopg import register_vector_async
from sqlalchemy import event, inspect, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


class DatabaseConfig:
    DBNAME = "colpalisearch"

    @classmethod
    def set_dbname(cls, name: str):
        cls.DBNAME = name


async_engine = create_async_engine(settings.database_url, echo_pool=True)
async_session = async_sessionmaker(bind=async_engine, expire_on_commit=False)


parsed_url = urlparse(settings.database_url)
conn_params = {
    "dbname": parsed_url.path.lstrip("/"),
    "user": parsed_url.username,
    "password": parsed_url.password,
    "host": parsed_url.hostname,
    "port": parsed_url.port,
}


@event.listens_for(async_engine.sync_engine, "connect")
def connect(dbapi_connection, connection_record):
    dbapi_connection.run_async(register_vector_async)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        connection = await session.connection()
        logger.info(
            f"Session in {get_session.__name__} has began: {connection.engine.url}"
        )
        yield session
        logger.info("Session closed")


async def get_table_names():
    async with async_engine.connect() as conn:
        table_names = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )
        return table_names


async def execute_analyze():
    table_names = await get_table_names()
    async with async_session() as session:
        for table_name in table_names:
            if table_name == "user":
                continue
            sql = text(f"ANALYZE {table_name};")
            await session.execute(sql)
