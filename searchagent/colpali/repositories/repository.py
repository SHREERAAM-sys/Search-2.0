from abc import ABC, abstractmethod

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession


class Repository[T](ABC):
    def __init__(self, model: T, session: AsyncSession):
        self.model = model
        self.session = session

    async def get_all(self) -> list[T]:
        async with self.session.begin():
            obj = await self.session.scalars(select(self.model))
            return obj.all()

    async def delete(self, t: T) -> None:
        delete_stm = delete(T).where(T.id == t.id)
        await self.session.execute(delete_stm)

    @abstractmethod
    async def add(self, **kwargs: object) -> T:
        raise NotImplementedError
