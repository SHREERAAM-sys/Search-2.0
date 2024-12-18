from itertools import chain
from typing import List, Tuple

from colpali_search.schemas.internal.pdf import ImageMetadata
from PIL import Image
from torch.utils.data import Dataset


class PDFImagesDataset(Dataset):
    def __init__(
        self,
        images: List[List[Image.Image]],
        metadata: List[List[ImageMetadata]],
    ):
        self.images = images
        self.metadata = metadata
        self.flat_images_metadata = self.flatten_images_metadata()

    def __len__(self):
        return len(self.flat_images_metadata)

    def flatten_images_metadata(self) -> List[Tuple[Image.Image, ImageMetadata]]:
        flattened_images = list(chain.from_iterable(self.images))
        flattened_metadata = list(chain.from_iterable(self.metadata))

        assert len(flattened_images) == len(flattened_metadata)

        return list(zip(flattened_images, flattened_metadata))

    def __getitem__(self, idx: int) -> Tuple[Image.Image, ImageMetadata]:
        image, metadata = self.flat_images_metadata[idx]
        return image, metadata
