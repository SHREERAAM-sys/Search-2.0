from datetime import datetime
from typing import Optional

import numpy as np
from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import ARRAY, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

"""
Check out the ERD Database design here:
https://drive.google.com/file/d/1AIpMmYtItZ8XGqRUUux1majA1Ue5sLSE/view?usp=sharing
"""

VECT_DIM = 128


class Base(DeclarativeBase):
    pass


class Page(Base):
    __tablename__ = "page"

    id: Mapped[int] = mapped_column(primary_key=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text_content: Mapped[Optional[str]] = mapped_column(String)
    binary_content: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    last_modified: Mapped[datetime] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(String, nullable=False)

    file_id: Mapped[int] = mapped_column(ForeignKey("file.id"))

    file: Mapped["File"] = relationship(back_populates="pages")
    embeddings: Mapped[list["Embedding"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Page(id={self.id}, page_number={self.page_number}, file_id={self.file_id})>"


class File(Base):
    __tablename__ = "file"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    filetype: Mapped[str] = mapped_column(String, nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(String)
    last_modified: Mapped[datetime] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(String, nullable=False)

    folder_id = mapped_column(ForeignKey("folder.id"), nullable=False)

    pages: Mapped[list[Page]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )
    folder: Mapped["Folder"] = relationship(back_populates="files")

    def __repr__(self) -> str:
        return (
            f"<File(id={self.id}, filename={self.filename}, filetype={self.filetype})>"
        )


class Embedding(Base):
    __tablename__ = "embedding"

    id: Mapped[int] = mapped_column(primary_key=True)
    vector_embedding: Mapped[list[np.array]] = mapped_column(ARRAY(HALFVEC(VECT_DIM)))
    last_modified: Mapped[datetime] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(String, nullable=False)

    page_id: Mapped[int] = mapped_column(ForeignKey("page.id"))

    page: Mapped[Page] = relationship(back_populates="embeddings")
    flattened_embeddings: Mapped[list["FlattenedEmbedding"]] = relationship(
        back_populates="embedding", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        return f"<Embedding(id={self.id}, page_id={self.page_id})>"


class FlattenedEmbedding(Base):
    __tablename__ = "flattened_embedding"

    id: Mapped[int] = mapped_column(primary_key=True)
    vector_embedding: Mapped[np.array] = mapped_column(HALFVEC(VECT_DIM))
    last_modified: Mapped[datetime] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(String, nullable=False)

    embedding_id: Mapped[int] = mapped_column(
        ForeignKey("embedding.id", ondelete="CASCADE")
    )

    embedding: Mapped[Embedding] = relationship(back_populates="flattened_embeddings")

    def __repr__(self) -> str:
        return f"<FlattenedEmbedding(id={self.id}, embedding_id={self.embedding_id})>"


class Folder(Base):
    __tablename__ = "folder"

    id: Mapped[int] = mapped_column(primary_key=True)
    folder_name: Mapped[str] = mapped_column(String, nullable=False)
    folder_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(String, nullable=False)

    user_id = mapped_column(ForeignKey("user.id"), nullable=False)

    user: Mapped["User"] = relationship(back_populates="folders")
    files: Mapped[list[File]] = relationship(
        back_populates="folder", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Folder(id={self.id}, folder_name={self.folder_name}, folder_path={self.folder_path})>"


class Query(Base):
    __tablename__ = "query"

    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    vector_embedding: Mapped[list[np.array]] = mapped_column(ARRAY(HALFVEC(VECT_DIM)))
    created_at: Mapped[datetime] = mapped_column(String, nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)

    user: Mapped["User"] = relationship(back_populates="queries")

    def __repr__(self) -> str:
        return f"<Query(id={self.id}, text={self.text})>"


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(String, nullable=False)

    queries: Mapped[list[Query]] = relationship(back_populates="user")
    folders: Mapped[list[Folder]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, created_at={self.created_at})>"
