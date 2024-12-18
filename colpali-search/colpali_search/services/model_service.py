from typing import Any, Callable, List, Optional, Tuple, Union

import ml_dtypes
import numpy as np
import torch
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from colpali_search.config import settings
from colpali_search.schemas.internal.pdf import ImageList, ImageMetadata, MetadataList
from colpali_search.services.pdf_images_dataset import PDFImagesDataset
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchFeature, PreTrainedModel


class ColPaliModelService:
    def __init__(self):
        """A RAG system using a visual language model called Colpali to process PDF files"""

        # Lazy loadings
        self._model = None
        self._processor: Optional[ColPaliProcessor] = None
        self.device = get_torch_device()
        self.model_name = settings.colpali_model_name
        self.hf_api_key = settings.hf_api_key

        if not self.hf_api_key:
            raise ValueError("HuggingFace API key is required")

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            self._model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                token=self.hf_api_key,
            ).eval()
        return self._model

    @property
    def processor(self) -> ColPaliProcessor:
        """Get the ColPaliProcessor instance"""
        if self._processor is None:
            self._processor = ColPaliProcessor.from_pretrained(
                self.model_name, token=self.hf_api_key
            )
        return self._processor

    @torch.no_grad()
    def image_collate_fn(
        self, batch: List[Tuple[Image.Image, ImageMetadata]]
    ) -> Tuple[BatchFeature, List[ImageMetadata]]:
        """Given a batch size of 4, it would return a list of 4 tuples and
        unpack the batch into a list of images

        `batch` before unpacking:
            [(Image1, ImageMetadata1), (Image2, ImageMetadata2), ...]

        `batch` after unpacking:
            [Image1, Image2, Image3, ...]
        """

        images, metadata = zip(*batch)

        images: List[Image.Image] = list(images)
        metadata: List[ImageMetadata] = list(metadata)

        batch_images = self.processor.process_images(images).to(self.device)
        return batch_images, metadata

    def _create_dataloader(
        self,
        dataset: Union[PDFImagesDataset, List[str]],
        process_fn: Union[
            Callable[[List[str]], torch.Tensor],
            Callable[
                [Tuple[Image.Image, ImageMetadata]],
                Tuple[BatchFeature, List[ImageMetadata]],
            ],
        ],
        batch_size: int = 4,  # 4 PDF image pages
    ) -> DataLoader[str]:
        if isinstance(dataset, List) and dataset and isinstance(dataset[0], str):
            dataset = ListDataset[str](dataset)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=process_fn,
        )

    def run_inference(self, batches: dict):
        return self.model(**batches)

    @torch.no_grad()
    def _embed(
        self,
        dataset: Union[PDFImagesDataset, List[str]],
        process_fn: Union[
            Callable[[List[str]], torch.Tensor],
            Callable[
                [List[Tuple[Image.Image, ImageMetadata]]],
                Tuple[BatchFeature, List[ImageMetadata]],
            ],
        ],
    ) -> List[torch.Tensor]:
        """Compute embeddings for either queries or images

        Args:
            dataset (Union[PDFImagesDataset, List[str]]): Images or queries to be processed
            process_fn (Union[
                Callable[[List[str]], torch.Tensor],
                Callable[
                    [List[Tuple[Image.Image, ImageMetadata]]],
                    Tuple[BatchFeature, List[ImageMetadata]],
                ]
            ):
                Either self.image_collate_fn or self.processor.process_queries

        Returns:
            List[torch.Tensor]: A list of computed embeddings
        """
        if process_fn not in [
            self.image_collate_fn,
            self.processor.process_queries,
        ]:
            raise ValueError(
                "process_fn must be either processor.process_images or processor.process_queries"
            )

        empty_lists = [[] for _ in range(2)]
        embeddings, _ = empty_lists

        dataloader = self._create_dataloader(dataset, process_fn)

        for batch in tqdm(dataloader):
            if isinstance(dataset, PDFImagesDataset):
                batch_images, _ = batch
                batches = {k: v.to(self.device) for k, v in batch_images.items()}
            else:
                batches = {k: v.to(self.device) for k, v in batch.items()}

            # Move images to GPU and get embeddings
            batch_embeddings = self.run_inference(batches)

            # Embeddings for a single PDF page; it is of the shape (1030, 128)
            # 32x32=1024 image patches and 6 instruction text tokens
            individual_embeddings = list(torch.unbind(batch_embeddings.to("cpu")))

            # Collect all embeddings
            embeddings.extend(individual_embeddings)

        return embeddings

    def embed_images(
        self, images: ImageList, pdf_metadata: MetadataList
    ) -> List[torch.Tensor]:
        """Embed images using custom Dataset
        Check this link on PyTorch custom Dataset:
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """

        processed_images = PDFImagesDataset(images, pdf_metadata)
        return self._embed(processed_images, self.image_collate_fn)

    def embed_query(self, query: str) -> List[NDArray[Any]]:
        qs = self._embed([query], self.processor.process_queries)

        # Based on https://github.com/pytorch/pytorch/issues/109873
        query_embeddings = [
            np.array(elem)
            for embed in qs
            for elem in embed.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        ]

        return query_embeddings

    def get_scores(self, qs: List[torch.Tensor], embeddings: List[torch.Tensor]):
        return self.processor.score(qs, embeddings)
