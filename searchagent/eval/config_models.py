from dataclasses import dataclass
from typing import List


@dataclass
class Dataset:
    source: str
    file_type: str
    format: List[str]


@dataclass
class LexicalSearch:
    method: str
    top_k: int


@dataclass
class SemanticSearch:
    method: str
    top_k: int
    options: List[str]


@dataclass
class Retrieval:
    strategy: str
    lexical: LexicalSearch
    semantic: SemanticSearch


@dataclass
class Reranking:
    model: str
    top_k: int


@dataclass
class RAGConfig:
    dataset: Dataset
    embedding_model: str
    retrieval: Retrieval
    reranking: Reranking


@dataclass
class SyntheticDatasetConfig:
    test_size: int
    file_type: str
    output_filename: str
    test_data_source: str
