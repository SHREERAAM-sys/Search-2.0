from colpali_search.database import get_session
from colpali_search.dependencies import IndexingServiceDep
from colpali_search.schemas.endpoints.index import (
    ConfigureIndexResponse,
    ResetIndexResponse,
    GetCurrentIndexStrategyResponse,
)
from colpali_search.custom_types import IndexingStrategyType
from colpali_search.utils import get_now
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

index_router = APIRouter(prefix="/index", tags=["index"])


@index_router.get("/")
async def list_supported_index_strategies():
    strategies = [
        {
            "name": "hnsw-cs",
            "summary": "HNSW with Cosine Similarity",
        },
        {
            "name": "exact",
            "summary": "Exact with MaxSim",
        },
        {
            "name": "hnsw-bq-hd",
            "summary": "HNSW with Binary Quantization and Hamming Distance",
        },
    ]
    return {"strategies": strategies}


@index_router.get("/current-strategy")
async def get_current_index_strategy(
    indexing_service: IndexingServiceDep, session: AsyncSession = Depends(get_session)
):
    """Retrieve the currently configured indexing strategy

    Args:
        indexing_service (IndexingServiceDep): Service for managing indexing strategies
        session (AsyncSession, optional): Database session for executing queries. Defaults to Depends(get_session)

    Raises:
        HTTPException: If the current indexing strategy cannot be retrieved

    Returns:
        GetCurrentIndexStrategyResponse: Details of the current indexing strategy, including its name and status
    """
    try:
        strategy = await indexing_service.get_current_strategy(session)
        name = strategy.strategy_name

        return GetCurrentIndexStrategyResponse(
            status="success", name=name, message=f"You are currently using {name}"
        )
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))


@index_router.post("/reset-strategy", description="Reset index strategy")
async def reset_index_strategy(
    indexing_service: IndexingServiceDep, session: AsyncSession = Depends(get_session)
):
    """Reset the current indexing strategy to its default configuration.

    Args:
        indexing_service (IndexingServiceDep): Service to manage indexing strategies.
        session (AsyncSession, optional): Database session for executing queries. Defaults to Depends(get_session).

    Raises:
        HTTPException: If resetting the strategy fails.

    Returns:
        ResetIndexResponse: Confirmation of the reset, including the reset time and status.
    """
    try:
        await indexing_service.reset_strategy(session)
        return ResetIndexResponse(
            status="success",
            message="Index strategy has been reset successfully.",
            reset_time=get_now(),
        )
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))


@index_router.post("/{strategy}", description="Configure an index strategy")
async def configure_index_strategy(
    strategy: IndexingStrategyType,
    indexing_service: IndexingServiceDep,
    session: AsyncSession = Depends(get_session),
):
    """Configure a specific index strategy for the search system

    Args:
        strategy (IndexingStrategyType): The desired indexing strategy to configure
        indexing_service (IndexingServiceDep): Service for managing indexing strategies
        session (AsyncSession, optional): Database session for executing queries. Defaults to Depends(get_session)

    Raises:
        HTTPException: If the provided strategy is invalid (422)
        HTTPException: If configuration fails (400)

    Returns:
        ConfigureIndexResponse: The configured indexing strategy details
    """
    try:
        print(strategy)
        if strategy == IndexingStrategyType.exact_maxsim:
            # FIXME: This is a hacky way to drop indexes if user wants to use exact search
            await indexing_service.drop_indexes(
                IndexingStrategyType.hnsw_cosine_similarity
            )
            return

        indexing_strategy = await indexing_service.configure_strategy(strategy, session)
        return ConfigureIndexResponse.model_validate(indexing_strategy)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
