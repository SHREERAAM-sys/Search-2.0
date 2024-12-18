from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class Metadata:
    filename: str
    filepath: str
    file_size: int
    created_at: datetime
    last_modified: datetime
    file_type: str


@dataclass
class Page:
    number: int
    content: str


@dataclass
class FileInfo:
    pages: List[Page]
    number: int
    metadata: Metadata
