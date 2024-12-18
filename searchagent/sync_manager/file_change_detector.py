from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

from searchagent.file_system.base import FileSystemManager
from searchagent.utils import get_now
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer


FileChangeEventList = List["FileChangeEvent"]


class FileChangeType(Enum):
    moved = "moved"
    modified = "modified"
    deleted = "deleted"
    created = "created"


@dataclass
class FileChangeEvent:
    filepath: str
    change_type: FileChangeType
    timestamp: datetime
    dest_path: Optional[str] = None


class Watcher:
    def __init__(self, path, event_list: FileChangeEventList) -> None:
        self.observer = Observer()
        self.path = path
        self.event_list = event_list

    def run(self):
        event_handler = EventHandler(self.event_list, self.path)
        self.observer.schedule(event_handler, self.path, recursive=True)
        self.observer.start()


class EventHandler(FileSystemEventHandler):
    def __init__(self, event_list: FileChangeEventList, path: str):
        self.event_list = event_list
        self.last_modified = datetime.now()
        self.allowed_extensions = FileSystemManager(dir=path).file_extension

    def create_event(
        self, event: FileSystemEvent, change_type: FileChangeType
    ) -> FileChangeEvent:
        dest_path = getattr(event, "dest_path", None)
        return FileChangeEvent(
            filepath=event.src_path,
            change_type=change_type,
            timestamp=get_now(),
            dest_path=dest_path,
        )

    def file_is_valid(self, event: FileSystemEvent) -> bool:
        ext = event.src_path.split(".")[-1]
        return not event.is_directory and ext in self.allowed_extensions

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        if self.file_is_valid(event):
            self.event_list.append(self.create_event(event, FileChangeType.created))

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
        if self.file_is_valid(event):
            self.event_list.append(self.create_event(event, FileChangeType.deleted))

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        if self.file_is_valid(event):
            if datetime.now() - self.last_modified < timedelta(seconds=1):
                return
            else:
                self.last_modified = datetime.now()
                self.event_list.append(
                    self.create_event(event, FileChangeType.modified)
                )

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        if self.file_is_valid(event):
            self.event_list.append(self.create_event(event, FileChangeType.moved))
