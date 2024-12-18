import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ml_dtypes
import numpy as np
import torch
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from PIL import Image
from searchagent.colpali.pdf_images_dataset import PDFImagesDataset
from searchagent.colpali.pdf_images_processor import PDFImagesProcessor
from searchagent.colpali.profiler import profile_colpali
from searchagent.colpali.schemas import ImageMetadata
from searchagent.colpali.search_engine.context import SearchContext
from searchagent.colpali.search_engine.strategy_factory import SearchStrategyFactory
from searchagent.colpali.service import EmbeddingSerivce
from searchagent.config import settings
from searchagent.db_connection import async_session
from searchagent.ragmetrics.metrics import measure_vram
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchFeature, PreTrainedModel
from loguru import logger


class ColPaliRag:
    def __init__(
        self,
        input_dir: Optional[Union[Path, str]] = None,
        benchmark_embed: bool = False,
        refresh: bool = False,
    ):
        """
        A RAG system using a visual language model called Colpali to process PDF files

        Args:
            input_dir (Optional[Union[Path, str]]): Path to the PDF directory
            benchmark_embed (bool): Allow us to benchmark the performance of ColPali RAG system on embedding
            refresh (bool): Flag indicating whether to embed documents
        """

        # Lazy loadings
        self._model = None
        self._processor: Optional[ColPaliProcessor] = None
        self.user_id = 1  # Replace with actual user ID
        self.metadata = None
        self.input_dir = None

        self.device = get_torch_device()
        self.model_name = settings.colpali_model_name
        self.hf_api_key = settings.hf_api_key
        self.benchmark_embed = benchmark_embed
        self.refresh = refresh

        if input_dir:
            self.input_dir = Path(input_dir)

        if not self.hf_api_key:
            raise ValueError("HuggingFace API key is required")

        if input_dir and not self.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        if input_dir and benchmark_embed:
            raise ValueError(
                "Benchmark mode is currently on. We will use ViDoRe dataset instead"
            )

        if input_dir:
            logger.info("Converting PDFs to images...")
            self.pdf_processor = PDFImagesProcessor.convert_pdf2image_from_dir(
                str(self.input_dir)
            )

        if not input_dir and benchmark_embed:
            self.pdf_processor = PDFImagesProcessor.retrieve_pdfImage_from_vidore()

        # Only convert PDFs to images and embed them when new docs are available
        if input_dir or benchmark_embed or refresh:
            self.images = self.pdf_processor.images_list
            self.pdf_metadata = self.pdf_processor.pdf_metadata

        self.embedding_service = EmbeddingSerivce(
            self.user_id, async_session(), self.input_dir, self.benchmark_embed
        )

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

    @profile_colpali(enable_profiling=False)
    def run_inference(self, batches: dict):
        return self.model(**batches)

    # @measure_latency_for_gpu(dummy_input_fn=lambda: create_dummy_input())
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
        embeddings, mdata = empty_lists

        dataloader = self._create_dataloader(dataset, process_fn)

        for batch in tqdm(dataloader):
            if isinstance(dataset, PDFImagesDataset):
                batch_images, metadata = batch
                batches = {k: v.to(self.device) for k, v in batch_images.items()}
                mdata.extend(metadata)
                self.metadata = mdata
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

    @measure_vram()
    def embed_images(self) -> List[torch.Tensor]:
        """Embed images using custom Dataset
        Check this link on PyTorch custom Dataset:
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
        processed_images = PDFImagesDataset(self.images, self.pdf_metadata)
        return self._embed(processed_images, self.image_collate_fn)

    @measure_vram()
    def embed_query(self, query: str) -> List[torch.Tensor]:
        return self._embed([query], self.processor.process_queries)

    def get_scores(self, qs: List[torch.Tensor], embeddings: List[torch.Tensor]):
        return self.processor.score(qs, embeddings)

    async def search(
        self, query: str, top_k: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """Search for the relevant file based on the query

        Args:
            query (str): Description about a file that user is searching
            top_k (int, optional): Top number of retrieved files. Defaults to 3.

        Returns:
            Optional[List[Dict[str, Any]]]: If the relevant file is found,
            it returns the page embeddings along with its metadata. Or None
        """

        # Run similarity search over the page embeddings for all the pages in the collection
        # top_indices has the shape of
        # tensor([[12,  0, 14],
        # [15, 14, 11]])
        qs = self.embed_query(query)

        # Based on https://github.com/pytorch/pytorch/issues/109873
        query_embeddings = [
            np.array(elem)
            for embed in qs
            for elem in embed.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        ]

        if self.benchmark_embed or self.refresh or self.input_dir:
            embeddings = await asyncio.to_thread(self.embed_images)

            await self.embedding_service.upsert_doc_embeddings(
                embeddings, self.metadata
            )

        await self.embedding_service.upsert_query_embeddings(query, query_embeddings)

        ctx = SearchContext(
            SearchStrategyFactory.create_strategy("ANNHNSWCosineSimilarity")
        )
        return await ctx.execute_search_strategy(query_embeddings, top_k)
