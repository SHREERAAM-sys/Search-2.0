import bcrypt
from colpali_search.database import async_session
from colpali_search.models import User
from colpali_search.utils import get_now
from loguru import logger
from pydantic import EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def add_new_user(session: AsyncSession, email: str, password: str) -> User:
    password_bytes = password.encode("utf-8")
    password_hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

    user_dict = {
        "email": email,
        "password_hash": password_hashed,
        "created_at": get_now(),
    }

    new_user = User(**user_dict)
    session.add(new_user)

    return new_user


async def seed_user_if_not_exists(email: EmailStr, password: str):
    async with async_session.begin() as session:
        connection = await session.connection()
        logger.info(
            f"Session in {seed_user_if_not_exists.__name__} has began: {connection.engine.url}"
        )
        stmt = select(User).where(User.email == email)

        result = await session.execute(stmt)
        existing_user = result.scalars().first()

        if existing_user:
            logger.info(f"User with email {email} already exists")
        else:
            logger.info(f"Seeding user with email {email} ...")
            return await add_new_user(session=session, email=email, password=password)
