from abc import ABC, abstractmethod

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class Repository[T](ABC):
    def __init__(self, model: T):
        self.model = model

    async def get_all(self, session: AsyncSession) -> list[T]:
        obj = await session.scalars(select(self.model))
        return obj.all()

    async def delete(self, id: int, session: AsyncSession) -> None:
        instance = await session.get(self.model, id)
        if instance:
            await session.delete(instance)
            await session.commit()

    @abstractmethod
    async def add(self, session: AsyncSession, **kwargs: object) -> T:
        raise NotImplementedError
