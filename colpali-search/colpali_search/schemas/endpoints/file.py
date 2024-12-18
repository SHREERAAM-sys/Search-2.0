import datetime
from typing import Optional

from colpali_search.custom_types import CustomBaseModel


class FileResultResponse(CustomBaseModel):
    id: int
    filename: str
    filetype: str
    total_pages: int
    summary: Optional[str]
    last_modified: datetime.datetime
    created_at: datetime.datetime
