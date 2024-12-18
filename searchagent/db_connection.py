from pgvector.psycopg import register_vector_async
from searchagent.config import settings
from sqlalchemy import event, inspect, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

DBNAME = "searchagent"

async_engine = create_async_engine(settings.database_url)
async_session = async_sessionmaker(bind=async_engine, expire_on_commit=False)


@event.listens_for(async_engine.sync_engine, "connect")
def connect(dbapi_connection, connection_record):
    dbapi_connection.run_async(register_vector_async)


# Enable the pgvector extension
async def enable_pgvector_extension():
    async with async_session() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


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
