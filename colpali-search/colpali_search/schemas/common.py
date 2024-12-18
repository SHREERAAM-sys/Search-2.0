from pydantic import BaseModel


class ImageMetadata(BaseModel):
    """Metadata on a single PDF Image"""

    filename: str
    page_number: int
    total_pages: int


class ErrorResponse(BaseModel):
    code: int
    message: str
