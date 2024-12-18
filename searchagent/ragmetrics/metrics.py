import time
import tracemalloc
from enum import Enum
from functools import lru_cache, wraps
from typing import List, Optional

import torch
from datasets import load_dataset
from searchagent.config import settings
from searchagent.ragmetrics.performance_logger import PerformanceLogger
from torchmetrics.retrieval import (
    RetrievalMAP,
    RetrievalMRR,
    RetrievalPrecision,
    RetrievalRecall,
)

ERROR_MSG = "No compatible GPU backend found (CUDA or MPS required)."
MB_DIVISOR = 1024**2


class MetricsType(Enum):
    recall = "recall"
    precision = "precision"
    mrr = "mrr"
    map = "map"

    def __str__(self) -> str:
        return self.value


metrics_function_mapping = {
    MetricsType.recall.value: RetrievalRecall,
    MetricsType.precision.value: RetrievalPrecision,
    MetricsType.mrr.value: RetrievalMRR,
    MetricsType.map.value: RetrievalMAP,
}


def get_torch_event():
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=True)
    elif torch.backends.mps.is_available():  # for Apple Silicon
        return torch.mps.Event(enable_timing=True)
    else:
        raise RuntimeError(ERROR_MSG)


def get_torch_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.mps.is_available():
        torch.mps.synchronize()
    else:
        raise RuntimeError(ERROR_MSG)


def measure_latency_for_gpu(dummy_input_fn, nb_iters=5, custom_msg=None):
    """Measure latency for GPU-bound tasks

    Args:
        dummy_input_fn (Callable): A function that returns the dummy input matching your production input shape
        nb_iters (int): Number of iterations to average the execution time
        custom_msg (str): User-defined message for custom log

    # Usage:
        @measure_latency_for_gpu(dummy_input_fn=lambda: create_dummy_input())
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # GPU warmup
            dummy_input = dummy_input_fn()

            for _ in range(5):
                _ = self.model(**dummy_input)

            start = get_torch_event()
            end = get_torch_event()

            start.record()
            for _ in range(nb_iters):
                result = func(self, *args, **kwargs)
            end.record()

            get_torch_synchronize()

            avg_latency = start.elapsed_time(end) / nb_iters

            PerformanceLogger().log(
                "gpu_latency",
                func.__name__,
                f"GPU Average Latency (over {nb_iters} iterations): {avg_latency:.3f} ms",
                custom_msg,
            )

            return result

        return wrapper

    return decorator


def measure_latency_for_cpu(custom_msg=None):
    """Measure latency for CPU-bound tasks"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()

            duration_ms = (end - start) * 1e3

            PerformanceLogger().log(
                "cpu_latency",
                func.__name__,
                f"CPU Latency: {duration_ms:.3f} ms",
                custom_msg,
            )

            return result

        return wrapper

    return decorator


def measure_ram(custom_msg=None):
    """Measure CPU memory"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracemalloc.start()

            try:
                result = func(*args, **kwargs)
                current, peak = tracemalloc.get_traced_memory()

                PerformanceLogger().log(
                    "ram_usage",
                    func.__name__,
                    f"RAM Usage - Current: {current / MB_DIVISOR:.2f} MB, Peak: {peak / MB_DIVISOR:.2f} MB",
                    custom_msg,
                )

                return result
            finally:
                tracemalloc.stop()

        return wrapper

    return decorator


def measure_vram(custom_msg=None):
    """Measure GPU memory"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=None)

            result = func(*args, **kwargs)

            if torch.mps.is_available():
                allocated_memory = torch.mps.current_allocated_memory()
                total_allocated_memory = torch.mps.driver_allocated_memory()
                value = (
                    f"VRAM currently allocated memory: {allocated_memory / MB_DIVISOR:.2f} MB | "
                    f"Total memory allocated by the driver: {total_allocated_memory / MB_DIVISOR:.2f} MB"
                )
            elif torch.cuda.is_available():
                vram_used = torch.cuda.max_memory_allocated(device=None)
                value = f"VRAM usage: {vram_used / MB_DIVISOR:.2f} MB"
            else:
                value = ERROR_MSG

            PerformanceLogger().log(
                "vram_usage",
                func.__name__,
                value,
                custom_msg,
            )

            return result

        return wrapper

    return decorator


def generate_target_tensor(top_k_indices: torch.Tensor) -> torch.Tensor:
    """Generate a target tensor with 1s where elements in top_k_indices
    match their row index, 0s otherwise

    Example:
        top_k_indices = tensor([[0, 3, 4],  # Query 0's top-k docs: [0, 3, 4]
                                [1, 2, 5]]) # Query 1's top-k docs: [1, 2, 5]

        result = generate_target_tensor(top_k_indices)
        # result: tensor([1, 0, 0, 1, 0, 0]) -> 1s for [0, 3] matching query IDs, 0 otherwise
    """
    row_indices = torch.arange(top_k_indices.size(0)).view(-1, 1)
    matches = (top_k_indices == row_indices).to(torch.int)
    return matches.flatten()


def generate_indexes_tensor(query_indices: List[int], top_k: int) -> torch.Tensor:
    """Generate row indices for each query

    Example:
        If top_k=5 and there are 2 queries,
        this will create a tensor like tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        where each number corresponds to the query ID for the top_k retrieved items
    """
    return torch.cat([torch.full((top_k,), i) for i in range(len(query_indices))])


@lru_cache(maxsize=3)
def fetch_dataset(dataset_size: Optional[int] = None):
    return load_dataset(
        settings.benchmark_dataset_name,
        split=f"test[:{dataset_size}]" if dataset_size else "test",
    )


def calculate_metrics_colpali(
    metrics: List[MetricsType],
    dataset_size: int,
    top_k: int,
):
    """Evaluate retrieval performance using specified metrics:

    - Precision: Measures the accuracy of retrieved relevant documents
    - Recall: Assesses the proportion of all relevant documents retrieved
    - MRR (Mean Reciprocal Rank): Reflects the rank position of the first relevant document
    - MAP (Mean Average Precision): Provides an overall measure of precision across recall levels

    Args:
        metrics (List[MetricsType]): List of metrics to calculate,
            e.g., [MetricsType.precision, MetricsType.recall]

        top_k (int, optional): Number of top results to consider. Defaults to TOP_K
        dataset (str, optional): The name of the Hugging Face dataset to be loaded
        dataset_size (int, optional): The number of queries to use for metric calculation.
        Loads the entire dataset if set to None
    """

    ds = fetch_dataset(dataset_size)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            scores = func(*args, **kwargs)
            top_k_scores, top_k_indices = torch.topk(scores, k=top_k, dim=1)

            preds = top_k_scores.flatten()
            query_indices = list(range(len(ds)))
            target = generate_target_tensor(top_k_indices)
            indexes = generate_indexes_tensor(query_indices, top_k)

            for metric in metrics:
                metric_function = metrics_function_mapping.get(metric)
                if metric_function:
                    m = metric_function()
                    percentage = m(preds, target, indexes=indexes)
                    result = percentage.item() * 1e2

                    PerformanceLogger().log(
                        f"{metric}",
                        func.__name__,
                        f"{metric}@{top_k}: {result:.2f}%",
                    )

        return wrapper

    return decorator


def fetch_dataset_column(
    column_name: str,
    dataset_size: Optional[int],
) -> List[int]:
    ds = fetch_dataset(dataset_size)
    return ds[column_name]


def recall(actual: List[int], predicted: List[int], k: int) -> float:
    """Measure proportion of relevant documents that are retrieved"""
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = round(len(act_set & pred_set) / float(len(act_set)), 2)
    return result


async def average_recall(
    rag, queries: List[str], actual_pages: List[int], k: int
) -> float:
    """Mean of recall scores across multiple queries"""
    total_recall = 0
    for query, page in zip(queries, actual_pages):
        response = await rag.search(query=query)
        predicted = [res["page_number"] for res in response]
        recall_score = recall([page], predicted, k)
        total_recall += recall_score
    average_recall = total_recall / len(queries)
    return average_recall


async def precision(rag, queries: List[str], actual_pages: List[int]):
    """Measure the proportion of retrieved documents that are relevant"""
    correct = 0
    for query, page in zip(queries, actual_pages):
        response = await rag.search(query=query)
        predicted = [res["page_number"] for res in response]
        if page in predicted:
            correct += 1
    return correct / len(queries)


async def mrr(rag, queries: List[str], actual_pages: List[int]):
    """Average the reciprocal ranks of the first relevant document retrieved across multiple queries"""
    total_rank = 0
    for query, page in zip(queries, actual_pages):
        response = await rag.search(query=query)
        predicted = [res["page_number"] for res in response]
        rank = 1 / (predicted.index(page) + 1) if page in predicted else 0
        total_rank += rank
    return total_rank / len(queries)
