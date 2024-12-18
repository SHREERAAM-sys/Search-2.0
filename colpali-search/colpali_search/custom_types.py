from enum import Enum
from typing import Any, List, TypeAlias

from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

VectorList: TypeAlias = List[NDArray]

QueryEmbeddingList: TypeAlias = List[NDArray[Any]]

alias_map = {
    "hnsw-cs": "HNSWCosineSimilarity",
    "exact": "ExactMaxSim",
    "hnsw-bq-hd": "HNSWBQHamming",
    "HNSWCosineSimilarity": "hnsw-cs",
    "ExactMaxSim": "exact",
    "HNSWBQHamming": "hnsw-bq-hd",
}


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)


class IndexingStrategyType(str, Enum):
    hnsw_cosine_similarity = "hnsw-cs"
    exact_maxsim = "exact"
    hnsw_binary_quantization_hamming_distance = "hnsw-bq-hd"

    @property
    def alias(self):
        try:
            return alias_map[self.value]
        except KeyError as e:
            raise ValueError(f"Invalid alias for {self.value}") from e

    @classmethod
    def from_alias(cls, alias: str):
        if alias not in alias_map:
            raise ValueError(f"'{alias}' is not a valid IndexingStrategyType alias")
        return cls(alias_map[alias])
