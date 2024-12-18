import pytest
import torch
from searchagent.colpali.base import ColPaliRag
from searchagent.utils import project_paths


@pytest.fixture
def load_colpali():
    colpali = ColPaliRag(input_dir=project_paths.SINGLE_FILE_DIR)
    query = "What are SigLIP-generated patch embeddings used for?"
    return colpali, query


def test_embed_query_return_type(load_colpali):
    colpali, query = load_colpali
    result = colpali.embed_query(query)
    assert isinstance(result, list)
    assert all(isinstance(item, torch.Tensor) for item in result)


def test_embed_images_dimensions(load_colpali):
    colpali, _ = load_colpali
    embeddings = colpali.embed_images()
    assert all(embed.shape == (1030, 128) for embed in embeddings)


def test_invalid_process_fn(load_colpali):
    def invalid_process_fn():
        return "This is invalid"

    colpali, _ = load_colpali
    dataset = ["s1", "s2", "s3"]
    with pytest.raises(ValueError):
        colpali._embed(dataset, invalid_process_fn)


def test_embeddings_by_page_id(load_colpali):
    colpali, _ = load_colpali
    colpali.embed_images()
    assert len(colpali.embeddings_by_page_id) > 0
    assert isinstance(colpali.stored_filepath, str)
