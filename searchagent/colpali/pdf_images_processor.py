import os
import tempfile
import uuid
from itertools import islice
from typing import Dict, List, Optional

from pdf2image import convert_from_path
from PIL import Image
from searchagent.colpali.schemas import ImageMetadata
from searchagent.file_system.base import FileSystemManager
from searchagent.ragmetrics.metrics import (
    fetch_dataset,
    measure_latency_for_cpu,
    measure_ram,
)


class PDFImagesProcessor:
    def __init__(
        self,
        input_dir: Optional[str] = None,
        images_list: Optional[List[List[Image.Image]]] = None,
        pdf_metadata: Optional[Dict[int, List[ImageMetadata]]] = None,
    ):
        self.input_dir = input_dir
        self.images_list = images_list
        self.pdf_metadata = pdf_metadata

    @staticmethod
    def convert_single_pdf(filepath: str) -> List[Image.Image]:
        with tempfile.TemporaryDirectory() as path:
            return convert_from_path(filepath, output_folder=path, thread_count=3)

    @staticmethod
    def generate_images_metadata(
        pdf_id: str,
        filepath: str,
        total_pages: int,
    ) -> List[ImageMetadata]:
        filename = os.path.basename(filepath)

        return [
            ImageMetadata(
                pdf_id=pdf_id,
                page_id=page_id + 1,  # Start at indx 1
                filename=filename,
                total_pages=total_pages,
                filepath=filepath,
            )
            for page_id in range(total_pages)
        ]

    @staticmethod
    def get_uuid() -> str:
        return str(uuid.uuid4())

    @classmethod
    @measure_latency_for_cpu()
    @measure_ram()
    def convert_pdf2image_from_dir(cls, input_dir: str, batch_size=4):
        filepaths = FileSystemManager(input_dir).list_files(["application/pdf"])

        images_list = []
        pdf_metadata: Dict[str, List[ImageMetadata]] = {}

        def process_batch(batch):
            for fp in batch:
                single_pdf_images = cls.convert_single_pdf(fp)
                total_pages = len(single_pdf_images)
                pdf_id = cls.get_uuid()

                images_list.append(single_pdf_images)
                pdf_metadata[pdf_id] = cls.generate_images_metadata(
                    pdf_id=pdf_id,
                    filepath=fp,
                    total_pages=total_pages,
                )

        it = iter(filepaths)
        while batch := list(islice(it, batch_size)):
            process_batch(batch)

        return cls(input_dir, images_list, pdf_metadata)

    @classmethod
    @measure_latency_for_cpu()
    @measure_ram()
    def retrieve_pdfImage_from_vidore(
        cls, dataset_size: Optional[int] = None, batch_size=4
    ):
        """We use this method to benchmark our rag system

        We will use the dataset from https://huggingface.co/datasets/vidore/syntheticDocQA_artificial_intelligence_test
        """
        ds = fetch_dataset(dataset_size=dataset_size)

        image_list = [[image] for image in ds["image"]]
        pdf_metadata: Dict[str, List[ImageMetadata]] = {}

        def process_batch(batch):
            for row in batch:
                pdf_id = cls.get_uuid()
                pdf_metadata[pdf_id] = [
                    ImageMetadata(
                        pdf_id=pdf_id,
                        page_id=int(row["page"]),
                        filename=row["image_filename"],
                        total_pages=1,
                        filepath="",
                    )
                ]

        it = iter(ds)
        while batch := list(islice(it, batch_size)):
            process_batch(batch)

        return cls(images_list=image_list, pdf_metadata=pdf_metadata)
