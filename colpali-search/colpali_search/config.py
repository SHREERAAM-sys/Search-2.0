from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    colpali_model_name: str = Field(
        default="vidore/colpali-v1.2",
        description="Colpali is a document retrieval model.",
    )
    benchmark_dataset_name: str = Field(
        default="vidore/syntheticDocQA_artificial_intelligence_test",
        description="This dataset includes documents about Artificial Intelligence.",
    )
    hf_api_key: str
    database_url: str
    test_database_url: str

    model_config = SettingsConfigDict(extra="allow")


settings = Settings()
