import fnmatch
import mimetypes
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional

import magic
from pydantic import BaseModel, field_validator
from searchagent.file_system.models import FileInfo, Page, Metadata
from searchagent.utils import timer


class FileSystemValidator(BaseModel):
    dir: str
    exclude_patterns: Optional[List[str]] = None

    @field_validator("dir")
    def validate_dir(cls, v):
        if not os.path.isdir(v):
            raise ValueError(f"The directory {v} is not valid.")
        return v

    @field_validator("exclude_patterns")
    def validate_patterns(cls, v):
        """Check if all patterns of 'v' are in allowable mimetypes"""
        if v is None:
            return []
        if all(pattern in list(mimetypes.types_map.keys()) for pattern in v):
            return v
        else:
            raise ValueError(f"One or more file extension patterns are invalid: {v}.")


class FileSystemManager:
    def __init__(self, dir: str, exclude_patterns: Optional[List[str]] = None):
        validated = FileSystemValidator(dir=dir, exclude_patterns=exclude_patterns)

        self.dir = validated.dir
        self.exclude_patterns = validated.exclude_patterns

        mimetypes.add_type(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".docx",
        )

    @property
    def file_extension(self):
        return [
            "pdf",  # PDF files
            "txt",  # Text files
            "doc",  # Doc files
            "docx",  # Microsoft Word files
            "pptx",  # PPTX files
            "csv",  # CSV files
            "xlsx",  # XLSX files
            "json",  # JSON files
            "yaml",  # YAML files
            "ipynb",  # Jupyter Notebooks
            "xml",  # XML files
            "md",  # Markdown files
            "py",  # Python files
            "java",  # Java files
            "js",  # JavaScript files
            "ts",  # TypeScript files
            "c",  # C files
            "cpp",  # C++ files
            "html",  # HTML files
        ]

    @property
    def retrievable_files(self):
        return [f".{ext}" for ext in self.file_extension]

    @timer
    def scan(self) -> List[FileInfo]:
        """
        Scan the entire directory and return a list of FileInfo objects

        E.g.

        fm = FileSystemManager(dir=dirname, exclude_patterns=["*.py"])
        out = fm.scan()

        Output:
            [
                FileInfo(content="...", metadata=Metadata(...)),
                FileInfo(content="...", metadata=Metadata(...)),
                FileInfo(content="...", metadata=Metadata(...)),
            ]
        """
        return list(self._yield_filepaths_recursively(self.dir))

    def _get_file_creation_time(self, stat: os.stat_result) -> float:
        """Getting file creation time across platforms

        Based on https://lwn.net/Articles/397442/:
        Linux systems do not store the file creation time,
        so we fall back to `st_mtime` (when the file was last modified)
        """
        my_os = platform.system()
        if my_os == "Windows" or "Darwin":
            return stat.st_birthtime
        else:
            return stat.st_mtime

    def _get_timestamp(self, timestamp: float) -> datetime:
        return datetime.fromtimestamp(timestamp, timezone.utc)

    def list_files(self, allowed_filetypes: Optional[List[str]]) -> List[str]:
        """List files from a directory"""
        return [
            entry.path
            for entry in os.scandir(self.dir)
            if entry.is_file()
            and magic.from_file(entry.path, mime=True) in allowed_filetypes
        ]

    def _yield_filepaths_recursively(self, dir: str) -> Generator[FileInfo, None, None]:
        # Based on https://docs.python.org/3/library/os.html#os.scandir,
        # `os.scandir` is more efficient for large datasets
        try:
            for entry in os.scandir(dir):
                filename = entry.name
                filepath = entry.path

                if not filename.startswith("."):
                    if entry.is_dir(follow_symlinks=False):
                        yield from self._yield_filepaths_recursively(filepath)
                    else:
                        if self._should_exclude(filepath):
                            continue

                        stat = entry.stat(follow_symlinks=False)

                        file_size = stat.st_size
                        created_at = self._get_timestamp(
                            self._get_file_creation_time(stat)
                        )
                        last_modified = self._get_timestamp(stat.st_mtime)
                        file_type = magic.from_file(entry.path, mime=True)
                        pages = self._construct_pages(filepath, file_type)

                        metadata = Metadata(
                            filename=filename,
                            filepath=filepath,
                            file_size=file_size,
                            created_at=created_at,
                            last_modified=last_modified,
                            file_type=file_type,
                        )

                        yield FileInfo(
                            pages=pages, number=len(pages), metadata=metadata
                        )
        except PermissionError:
            pass

    def _construct_pages(self, filepath: str, file_type: str) -> List[Page]:
        if file_type == mimetypes.types_map[".pdf"]:
            return self.extract_text_from_pdf(filepath)
        if file_type == mimetypes.types_map[".docx"]:
            return self.extract_text_from_docx(filepath)
        else:
            return [Page(1, content=Path(filepath).read_text())]

    def _to_be_excluded(self, filepath: str, patterns: List[str]) -> bool:
        return any(fnmatch.fnmatch(filepath, f"*{ext}") for ext in patterns)

    def _should_exclude(self, filepath: str) -> List[FileInfo]:
        """If any of the patterns match the filepath, the file will be excluded"""
        is_retrievable = self._to_be_excluded(filepath, self.retrievable_files)
        is_excluded = self._to_be_excluded(filepath, self.exclude_patterns)
        return not is_retrievable or is_excluded

    def extract_text_from_pdf(self, filepath: str) -> List[Page]:
        from pypdf import PdfReader

        reader = PdfReader(filepath)
        rpages = reader.pages
        return [
            Page(number=i + 1, content=rpages[i].extract_text())
            for i in range(len(rpages))
        ]
