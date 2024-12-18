import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, model_validator


class SearchResult(BaseModel):
    file_id: int
    page_number: int
    filename: str
    filetype: str
    total_pages: int
    summary: Optional[str]
    last_modified: datetime.datetime
    created_at: datetime.datetime


class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, description="The search query must not be empty."
    )
    top_k: int = Field(..., gt=0, description="top_k must be greater than 0.")
    email: EmailStr

    @model_validator(mode="before")
    def validate_query_and_top_k(cls, values):
        query = values.get("query")
        top_k = values.get("top_k")

        if not query or query.strip() == "":
            raise ValueError("Query cannot be empty or only whitespace.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        return values


class SearchResponse(BaseModel):
    result: List[SearchResult]


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str


class CreateUserResponse(BaseModel):
    status: str
    email: EmailStr
    message: str
