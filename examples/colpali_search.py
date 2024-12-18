from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from searchagent.colpali.base import ColPaliRag
from searchagent.db_connection import enable_pgvector_extension
from searchagent.ragmetrics.metrics import (
    average_recall,
    fetch_dataset_column,
    mrr,
    precision,
)
from searchagent.utils import project_paths

pdfs_dir = project_paths.PDF_DIR
single_file_dir = project_paths.SINGLE_FILE_DIR

# Load benchmark dataset
dataset_size = 100
pages = fetch_dataset_column(column_name="page", dataset_size=dataset_size)
queries = fetch_dataset_column(column_name="query", dataset_size=dataset_size)
actual_pages = [int(page_number) for page_number in pages]


rag = ColPaliRag()
_ = rag.model
_ = rag.processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    await enable_pgvector_extension()
    logger.info("Starting up")
    yield
    logger.info("Shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "ColPali Search API", "docs": "/docs", "health": "/health"}


@app.post("/search")
async def search(query: str):
    response = await rag.search(query=query)
    return {"response": response}


@app.post("/benchmark")
async def benchmark(top_k: int):
    average_recall_score = await average_recall(rag, queries, actual_pages, top_k)
    precision_score = await precision(rag, queries, actual_pages)
    mrr_score = await mrr(rag, queries, actual_pages)

    return {
        "average_recall": average_recall_score,
        "precision": precision_score,
        "mrr": mrr_score,
    }
