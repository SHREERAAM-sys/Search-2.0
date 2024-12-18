from typing import List, TypeAlias

from colpali_search.schemas.common import ImageMetadata
from colpali_search.custom_types import CustomBaseModel
from PIL import Image

ImageList: TypeAlias = List[List[Image.Image]]

MetadataList: TypeAlias = List[List[ImageMetadata]]


class SinglePDFConversion(CustomBaseModel):
    single_pdf_images: List[Image.Image]
    metadata: List[ImageMetadata]


class PDFsConversion(CustomBaseModel):
    images_list: ImageList
    metadata_list: MetadataList
