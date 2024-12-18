import os

import pytest
from searchagent.colpali.base import ColPaliRag
from searchagent.utils import project_paths
import torch


def test_valid_output():
    query = "Tell me about the latencies and memory footprint for online query"
    colpali = ColPaliRag(input_dir=project_paths.SINGLE_FILE_DIR)
    results = colpali.search(query, top_k=3)

    assert isinstance(results, list)
    for result in results:
        assert isinstance(result["embedding"], torch.Tensor)
        assert isinstance(result["metadata"], dict)
        assert isinstance(result["created_at"], str)
        assert isinstance(result["modified_at"], str)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("What is the architecture of Late interaction based Vision Retrieval?", 5),
        ("What does it say about Unstructured when assessing current system", 4),
        ("Tell me about ColPali's performance", 6),
        ("What are the conclusions on ColPali", 8),
    ],
)
def test_single_pdf(query, expected):
    colpali = ColPaliRag(input_dir=project_paths.SINGLE_FILE_DIR)
    filepath = f"{os.getcwd()}/embeddings_metadata.json"
    results = colpali.search(query, top_k=3, filepath=filepath)
    page_ids = [res["metadata"]["page_id"] for res in results]

    assert expected in page_ids


@pytest.mark.parametrize(
    "query, expected",
    [
        (
            "What are Position-wise Feed-Forward Networks?",
            "Attention_Is_All_You_Need.pdf",
        ),
        (
            "What is a novel technique for visual encoding of videos "
            "to generate semantically rich captions?",
            "Aafaq_Spatio-Temporal_Dynamics_and_Semantic_Attribute_"
            "Enriched_Visual_Encoding_for_Video_CVPR_2019_paper.pdf",
        ),
        (
            "What is the effect of unimodal improvements on multi-modal fusion?",
            "Abavisani_Improving_the_Performance_of_Unimodal_Dynamic_"
            "Hand-Gesture_Recognition_With_Multimodal_CVPR_2019_paper.pdf",
        ),
    ],
)
def test_multi_pdfs(query, expected):
    colpali = ColPaliRag(input_dir=project_paths.PDF_DIR)
    results = colpali.search(query, top_k=1)
    filename = [res["metadata"]["filename"] for res in results]

    assert expected in filename
