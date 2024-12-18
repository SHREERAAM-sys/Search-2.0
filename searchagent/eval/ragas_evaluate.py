import argparse
import json
import os
import time
from typing import List

import yaml
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from loguru import logger
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestDataset, TestsetGenerator
from searchagent.eval.config_models import RAGConfig, SyntheticDatasetConfig
from searchagent.eval.utils import (
    get_file_path,
    is_file_in_directory,
    is_yaml_or_json_file,
)
from searchagent.utils import project_paths

load_dotenv()


class RAGASEvaluate:
    def __init__(self) -> None:
        self.default_rag_config_file = "default_config.yaml"
        self.default_dataset_config_file = "default_synthetic_dataset_settings.json"
        self.config_rag_path, self.config_dataset_path = self.get_config_path()
        self.config = self.load_rag_config()

    @property
    def llm_model(self) -> str:
        return "llama3.1:8b-instruct-fp16"

    @property
    def embed_model(self) -> str:
        return "sentence-transformers/all-MiniLM-l6-v2"

    def get_config_path(self) -> str | None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c",
            "--rag_config",
            default=self.default_rag_config_file,
            help="Path to the configuration file",
        )

        parser.add_argument(
            "-s",
            "--dataset_config",
            default=self.default_dataset_config_file,
            help="Path to the synthetic dataset configuration file",
        )

        try:
            args = parser.parse_args()
            return self.validate_config_path(
                args.rag_config
            ), self.validate_config_path(args.dataset_config)
        except Exception as e:
            logger.error(f"Unexpected error while parsing arguments: {e}")
            return None

    def validate_config_path(self, filename: str) -> str | None:
        directory = project_paths.EVAL_CONFIG
        file_path = get_file_path(filename, directory)

        if not is_yaml_or_json_file(file_path):
            logger.error(f"File is not a valid YAML file: {file_path}")
            return None

        if not is_file_in_directory(file_path):
            logger.error(f"File is not in the specified directory: {file_path}")
            return None

        logger.info(f"Using config file: {file_path}")

        return file_path

    def load_rag_config(self) -> RAGConfig:
        try:
            with open(self.config_rag_path, "r") as file:
                data = yaml.safe_load(file)
                return RAGConfig(**data)
        except Exception as e:
            logger.error(f"Error loading RAG configuration: {e}")

    @classmethod
    def load_dataset_config(cls, file_path: str) -> SyntheticDatasetConfig:
        try:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                return SyntheticDatasetConfig(**data)
        except Exception as e:
            logger.error(f"Error loading configuration for synthetic dataset: {e}")

    def setup_models(self) -> tuple[OllamaLLM, CacheBackedEmbeddings]:
        HF_API_KEY = os.getenv("HF_API_KEY")
        llm = OllamaLLM(model=self.llm_model)
        underlying_embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_API_KEY,
            model_name=self.embed_model,
        )
        store = LocalFileStore("./cache/")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            store,
            namespace=underlying_embeddings.model_name,
        )

        return llm, cached_embedder

    @staticmethod
    def get_data_loader(file_type: str):
        if file_type == "pdf":
            return DirectoryLoader
        else:
            raise ValueError(f"Unsupported file type: ${file_type}")

    @staticmethod
    def save_synthetic_test_data(testset: TestDataset, output_filename: str):
        OUTPUT_TESTDATA_DIR = project_paths.DATA / "synthetic_testdata"

        if not os.path.exists(OUTPUT_TESTDATA_DIR):
            os.mkdir(OUTPUT_TESTDATA_DIR)

        df = testset.to_pandas()
        df.to_csv(OUTPUT_TESTDATA_DIR / output_filename, index=False)

    @classmethod
    def load_documents(cls, data_source: str, file_type: str) -> List[Document]:
        loader_class = cls.get_data_loader(file_type)
        loader = loader_class(data_source)
        return loader.load()

    def generate_synthetic_test_data(self):
        config = self.load_dataset_config(self.config_dataset_path)
        docs = self.load_documents(config.test_data_source, config.file_type)
        llm, cached_embeddings = self.setup_models()

        generator = TestsetGenerator.from_langchain(
            generator_llm=llm, critic_llm=llm, embeddings=cached_embeddings
        )

        distributions = {simple: 0.3, reasoning: 0.4, multi_context: 0.3}

        logger.info("Starting test data generation")
        start_time = time.time()

        testset = generator.generate_with_langchain_docs(
            docs,
            test_size=config.test_size,
            distributions=distributions,
        )

        end_time = time.time()
        logger.info(f"Test data generation took {end_time - start_time:.2f} seconds")

        self.save_synthetic_test_data(testset, config.output_filename)


# Usage
rag_evaluate = RAGASEvaluate()
rag_evaluate.generate_synthetic_test_data()
