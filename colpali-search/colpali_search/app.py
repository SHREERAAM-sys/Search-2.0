import sys
from contextlib import asynccontextmanager

from colpali_search.context import app_context, initialize_context
from colpali_search.database import get_session
from colpali_search.dependencies import BenchmarkServiceDep, SearchSerivceDep
from colpali_search.models import User
from colpali_search.routers import embeddings, files, index
from colpali_search.schemas.endpoints.benchmark import BenchmarkResponse
from colpali_search.schemas.endpoints.search import (
    CreateUserRequest,
    CreateUserResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from colpali_search.seed_user import seed_user_if_not_exists
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger.remove()
logger.add(sys.stderr, colorize=True)

description = """
ColPali Search API for searching local PDF files using natural language. 🚀

## Default

You can **search local files** or **benchmark the RAG system**.


## Files
You will be able to:

* **Get all files**
* **Get file by id**
* **Delete file by id**


## Embeddings

You will be able to:

* **Generate embeddings for file**
* **Generate embeddings for files**
* **Generate embeddings for benchmarking**


## Index

You will be able to:

* **List all supported index strategies**
* **Configure index strategy**
* **Reset index strategy**
* **Get current index strategy**
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_context()
    _ = app_context.model_service.model
    _ = app_context.model_service.processor
    yield
    app_context.model_service = None
    app_context.embedding_service = None
    app_context.pdf_conversion_service = None
    app_context.search_service = None


app = FastAPI(
    lifespan=lifespan,
    title="ColPali Search",
    description=description,
    version="0.0.1",
    contact={
        "name": "Jingying Huang",
        "email": "jingyingfhuang@gmail.com",
    },
)
api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(embeddings.embeddings_router)
api_v1_router.include_router(files.files_router)
api_v1_router.include_router(index.index_router)


async def get_current_user(
    email: str = "colpalisearch@gmail.com",
    session: AsyncSession = Depends(get_session),
) -> int:
    """Retrieve current user from the database

    Args:
        email (str, optional): Defaults to "colpalisearch@gmail.com"
        session (AsyncSession, optional): Defaults to Depends(get_session)

    Raises:
        HTTPException: Return 404 if user is not found

    Returns:
        int: a unique user id
    """
    stmt = select(User).where(User.email == email)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.id


@api_v1_router.post("/search")
async def search(
    body: SearchRequest,
    search_service: SearchSerivceDep,
    user_id: int = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> SearchResponse:
    """Search for specific PDF files given a query

    Args:
        body (SearchRequest): Request includes query, top_k, and email
        search_service (SearchSerivceDep): Search service dependency
        user_id (int, optional): Defaults to Depends(get_current_user)
        session (AsyncSession, optional): Defaults to Depends(get_session)

    Raises:
        HTTPException: Throw exception if user id is invalid

    Returns:
        SearchResponse: Contain retrieved file info such as filename, total_pages, etc.
    """
    logger.info(
        f"Received search request from user_id={user_id} for query='{body.query}'"
    )

    try:
        result = await search_service.search(user_id, body.query, body.top_k, session)
        return SearchResponse(result=[SearchResult(**item) for item in result])

    except Exception as e:
        logger.error(f"Error during search for user_id={user_id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@api_v1_router.post("/benchmark")
async def benchmark(
    top_k: int,
    benchmark_service: BenchmarkServiceDep,
    user_id: int = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> BenchmarkResponse:
    """Benchmark the current RAG system using vidore/syntheticDocQA_artificial_intelligence_test

    Args:
        top_k (int): Define number of retrieved documents
        benchmark_service (BenchmarkServiceDep): Benchmark service dependency
        user_id (int, optional): Defaults to Depends(get_current_user)
        session (AsyncSession, optional): Defaults to Depends(get_session)

    Returns:
        BenchmarkResponse: Return average recall score, precision score, and mrr score
    """
    average_recall_score = await benchmark_service.average_recall(
        top_k, user_id, session
    )
    precision_score = await benchmark_service.precision(top_k, user_id, session)
    mrr_score = await benchmark_service.mrr(top_k, user_id, session)

    return BenchmarkResponse(
        average_recall_score=average_recall_score,
        precision_score=precision_score,
        mrr_score=mrr_score,
    )


@app.post("/user/create")
async def create_user(body: CreateUserRequest):
    """Creates a new user in the system.

    This endpoint allows clients to create a new user by providing an email
    and password. If the user already exists, a success message is returned
    with the existing user information.

    Args:
        body (CreateUserRequest): The request body containing the user's email
        and password.

    Raises:
        HTTPException: If an unexpected error occurs during user creation,
        a 500 Internal Server Error is returned.

    Returns:
        CreateUserResponse: A response containing the status, email of the
        created user, and a success message.
    """
    try:
        user = await seed_user_if_not_exists(body.email, body.password)
        email = user.email
        return CreateUserResponse(
            status="success",
            email=email,
            message=f"You have successfully created a user with email: {email}",
        )
    except Exception as e:
        logger.error(f"Error when creating a new user: {body.email}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.get("/")
async def info():
    return {
        "service_name": "Search2.0 API",
        "version": "1.0.0",
        "description": "API for searching local files using natural language",
    }


app.include_router(api_v1_router)
