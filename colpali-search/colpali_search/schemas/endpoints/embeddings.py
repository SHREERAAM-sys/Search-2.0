from typing import List

from colpali_search.schemas.common import ImageMetadata
from colpali_search.custom_types import CustomBaseModel


class EmbeddingsResponse(CustomBaseModel):
    message: str
    embeddings: List[List[float]]
    metadata: List[ImageMetadata]
