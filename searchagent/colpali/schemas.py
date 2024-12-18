from __future__ import annotations

from pydantic import BaseModel


class ImageMetadata(BaseModel):
    """Metadata on a single PDF Image"""

    pdf_id: str
    page_id: int
    filename: str
    total_pages: int
    filepath: str
