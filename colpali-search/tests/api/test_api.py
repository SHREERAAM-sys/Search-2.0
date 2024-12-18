from pathlib import Path

import pytest
import pytest_asyncio
from colpali_search.app import app
from httpx import ASGITransport, AsyncClient

BASE_PATH = Path(__file__).parent.parent

PDFS_DIR = BASE_PATH / "pdfs"
INVALID_FILE_DIR = BASE_PATH / "invalid_files"


@pytest_asyncio.fixture
async def async_client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def setup_dummy_files(async_client):
    files = []
    filenames = [
        "simd.pdf",
        "improving_efficient_neural_ranking.pdf",
        "Attention_Is_All_You_Need.pdf",
    ]

    for filename in filenames:
        test_file_path = PDFS_DIR / filename
        assert test_file_path.exists(), f"Test file not found: {test_file_path}"
        files.append(("files", open(test_file_path, "rb")))

    response = await async_client.post("/api/v1/embeddings/files", files=files)

    # Close files after uploading to avoid resource leaks
    for file in files:
        file[1].close()

    assert response.status_code == 200
    return response


@pytest.mark.anyio
async def test_embeddings_file(async_client):
    test_file_path = PDFS_DIR / "simd.pdf"

    assert test_file_path.exists(), f"Test file not found: {test_file_path}"

    with open(test_file_path, "rb") as file:
        response = await async_client.post(
            "/api/v1/embeddings/file", files={"file": file}
        )

    assert response.status_code == 200
    data = response.json()

    assert data["message"] == "File successfully embedded"
    assert data["metadata"][0]["filename"] == "simd.pdf"
    assert data["metadata"][0]["total_pages"] == 12


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filepath, status_code",
    [
        (PDFS_DIR / "simd.pdf", 200),
        ("", 422),
        (INVALID_FILE_DIR / "LICENSE.txt", 404),
        (INVALID_FILE_DIR / "cybertruck.png", 404),
    ],
)
async def test_embeddings_edge_cases(async_client, filepath, status_code):
    files = {} if filepath == "" else {"file": open(filepath, "rb")}
    response = await async_client.post("/api/v1/embeddings/file", files=files)
    assert response.status_code == status_code


@pytest.mark.anyio
async def test_embeddings_files(async_client, setup_dummy_files):
    data = setup_dummy_files.json()

    assert len(data["metadata"]) == 37

    # Each page has a shape of (1030, 128)
    assert len(data["embeddings"][0]) / 1030 == 128


@pytest.mark.anyio
async def test_search(async_client, setup_dummy_files):
    query = "Find me a paper on attention mechanism"

    search_request = {"query": query, "top_k": 2, "email": "colpalisearch@gmail.com"}

    # Include the query in the URL
    response = await async_client.post("/api/v1/search", json=search_request)
    data = response.json()

    assert data["result"][0]["filename"] == "Attention_Is_All_You_Need.pdf"
    assert data["result"][0]["total_pages"] == 15


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,top_k,status_code",
    [
        ("valid query", 5, 200),
        ("", 5, 422),  # Empty query
        ("valid query", 0, 422),  # Invalid top_k
    ],
)
async def test_search_edge_cases(async_client, query, top_k, status_code):
    response = await async_client.post(
        "/api/v1/search",
        json={"query": query, "top_k": top_k, "email": "test@example.com"},
    )
    assert response.status_code == status_code


@pytest.mark.anyio
async def test_get_all_files(async_client, setup_dummy_files):
    response = await async_client.get("/api/v1/files/")

    assert response.status_code == 200
    assert len(response.json()) == 3


@pytest.mark.anyio
async def test_get_file_by_id(async_client, setup_dummy_files):
    response = await async_client.get("/api/v1/files/1")
    data = response.json()

    assert data["filename"] == "simd.pdf"
    assert data["total_pages"] == 12


@pytest.mark.asyncio
@pytest.mark.xfail
async def test_get_file_by_invalid_id(async_client, setup_dummy_files):
    response = await async_client.get("/api/v1/files/4")
    assert response.status_code == 500


@pytest.mark.anyio
async def test_delete_file_by_id(async_client, setup_dummy_files):
    delete_response = await async_client.delete("/api/v1/files/2")
    assert delete_response.json() == {"success": "Deleted"}

    files_response = await async_client.get("/api/v1/files/")
    files_data = files_response.json()

    assert len(files_data) == 2
    assert files_data[0]["filename"] == "simd.pdf"
    assert files_data[1]["filename"] == "Attention_Is_All_You_Need.pdf"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "strategy, expected_status, expected_index_name",
    [
        ("hnsw-cs", 200, "HNSWCosineSimilarity"),
        ("hnsw-bq-hd", 200, "HNSWBQHamming"),
        ("exact", 200, None),
        ("invalid_strategy", 422, None),  # Invalid strategy
    ],
)
async def test_configure_index_strategy(
    async_client, strategy, expected_status, expected_index_name
):
    response = await async_client.post(f"/api/v1/index/{strategy}")
    data = response.json()

    assert response.status_code == expected_status

    if response.status_code == 200 and strategy != "exact":
        assert data["strategy_name"] == expected_index_name

    if strategy == "exact":
        assert data is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "strategy, expected_strategy",
    [
        ("hnsw-cs", "HNSWCosineSimilarity"),
        ("exact", None),
        ("hnsw-bq-hd", "HNSWBQHamming"),
    ],
)
async def test_get_current_strategy_hnsw_cs(async_client, strategy, expected_strategy):
    await async_client.post(f"/api/v1/index/{strategy}")

    response = await async_client.get("/api/v1/index/current-strategy")
    data = response.json()

    if strategy != "exact":
        assert data["status"] == "success"
        assert data["name"] == expected_strategy


@pytest.mark.anyio
async def test_reset_index_strategy(async_client):
    response = await async_client.post("/api/v1/index/reset-strategy")
    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "success"

    current_strategy_response = await async_client.get("/api/v1/index/current-strategy")
    current_strategy_data = current_strategy_response.json()

    assert current_strategy_data["name"] == "HNSWCosineSimilarity"
