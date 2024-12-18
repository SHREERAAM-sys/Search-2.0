from typing import List

from colpali_search.models import File
from colpali_search.repository.repositories import FileRepository
from sqlalchemy.ext.asyncio import AsyncSession


class FileService:
    def __init__(self):
        self.file_repository = FileRepository(File)

    async def get_file_by_id(self, id: int, session: AsyncSession) -> File:
        return await self.file_repository.get_by_id(id, session)

    async def get_all_files(self, session: AsyncSession) -> List[File]:
        return await self.file_repository.get_all(session)

    async def delete_file_by_id(self, id: int, session: AsyncSession):
        return await self.file_repository.delete(id, session)
