from __future__ import annotations

import asyncio
from itertools import islice
from typing import List, Optional

from colpali_search.schemas.internal.pdf import (
    ImageList,
    ImageMetadata,
    MetadataList,
    PDFsConversion,
    SinglePDFConversion,
)
from colpali_search.services.benchmark_service import BenchmarkService
from fastapi import UploadFile
from pdf2image import convert_from_bytes


class PDFConversionService:
    def __init__(self, benchmark_service: BenchmarkService):
        self.benchmark_service = benchmark_service

    def _generate_images_metadata(
        self,
        filename: str,
        total_pages: int,
    ) -> List[ImageMetadata]:

        return [
            ImageMetadata(
                page_number=page_number + 1,
                filename=filename,
                total_pages=total_pages,
            )
            for page_number in range(total_pages)
        ]

    def convert_single_pdf2image(self, pdf_file: UploadFile) -> SinglePDFConversion:
        bytes = asyncio.run(pdf_file.read())
        single_pdf_images = convert_from_bytes(
            bytes, thread_count=3, use_pdftocairo=True, fmt="jpeg"
        )
        total_pages = len(single_pdf_images)
        metadata = self._generate_images_metadata(
            filename=pdf_file.filename,
            total_pages=total_pages,
        )
        return SinglePDFConversion(
            single_pdf_images=single_pdf_images, metadata=metadata
        )

    def convert_pdfs2image(
        self, pdf_files: List[UploadFile], batch_size=4
    ) -> PDFsConversion:
        images_list: ImageList = []
        metadata_list: MetadataList = []

        def process_batch(batch):
            for pdf_file in batch:
                result = self.convert_single_pdf2image(pdf_file)
                images_list.append(result.single_pdf_images)
                metadata_list.append(result.metadata)

        it = iter(pdf_files)
        while batch := list(islice(it, batch_size)):
            process_batch(batch)

        return PDFsConversion(images_list=images_list, metadata_list=metadata_list)

    def retrieve_pdfImage_from_vidore(
        self, dataset_size: Optional[int] = None, batch_size=4
    ):
        """We use this method to benchmark our rag system

        We will use the dataset from https://huggingface.co/datasets/vidore/syntheticDocQA_artificial_intelligence_test
        """
        ds = self.benchmark_service.fetch_dataset(dataset_size=dataset_size)

        images_list = [[image] for image in ds["image"]]
        metadata_list: MetadataList = []

        def process_batch(batch):
            for row in batch:
                metadata_list.append(
                    [
                        ImageMetadata(
                            filename=row["image_filename"],
                            page_number=int(row["page"]),
                            total_pages=1,
                        )
                    ]
                )

        it = iter(ds)
        while batch := list(islice(it, batch_size)):
            process_batch(batch)

        return PDFsConversion(images_list=images_list, metadata_list=metadata_list)
