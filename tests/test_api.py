from unittest.mock import patch

from app.exceptions import IngestionError, LLMError

# --- Health -----------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# --- /repos -----------------------------------------------------------------


def test_repos_empty(client):
    r = client.get("/repos")
    assert r.status_code == 200
    assert r.json() == {"repos": []}


def test_repos_lists_after_ingest(client, sample_repo_path):
    with patch("app.main.clone_repo", return_value=sample_repo_path):
        client.post("/ingest", json={"repo_url": "https://github.com/x/y"})

    r = client.get("/repos")
    assert r.status_code == 200
    assert "sample_repo" in r.json()["repos"]


# --- /ingest ----------------------------------------------------------------


def test_ingest_success(client, sample_repo_path):
    with patch("app.main.clone_repo", return_value=sample_repo_path):
        r = client.post("/ingest", json={"repo_url": "https://github.com/x/y"})

    assert r.status_code == 200
    body = r.json()
    assert body["repo_name"] == "sample_repo"
    assert body["chunks_created"] > 0
    assert body["files_indexed"] > 0
    assert body["status"] == "ok"


def test_ingest_propagates_ingestion_error(client):
    with patch(
        "app.main.clone_repo",
        side_effect=IngestionError("clone failed", status_code=400),
    ):
        r = client.post("/ingest", json={"repo_url": "https://github.com/x/y"})

    assert r.status_code == 400
    assert "clone failed" in r.json()["detail"]


# --- /query -----------------------------------------------------------------


def test_query_no_repos_returns_400(client):
    r = client.post("/query", json={"question": "hi"})
    assert r.status_code == 400
    assert "No repos" in r.json()["detail"]


def test_query_success(client, sample_repo_path):
    with (
        patch("app.main.clone_repo", return_value=sample_repo_path),
        patch("app.utilities.query_utils._run_llm", return_value="mocked answer"),
    ):
        client.post("/ingest", json={"repo_url": "https://github.com/x/y"})
        r = client.post("/query", json={"question": "How does auth work?"})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "mocked answer"
    assert isinstance(body["citations"], list)
    assert body["retrieved_chunks"]


def test_query_unknown_repo_returns_404(client, sample_repo_path):
    with patch("app.main.clone_repo", return_value=sample_repo_path):
        client.post("/ingest", json={"repo_url": "https://github.com/x/y"})

    r = client.post(
        "/query",
        json={"question": "hi", "repo_name": "nonexistent"},
    )
    assert r.status_code == 404


def test_query_llm_error_returns_mapped_status(client, sample_repo_path):
    with (
        patch("app.main.clone_repo", return_value=sample_repo_path),
        patch(
            "app.utilities.query_utils._run_llm",
            side_effect=LLMError("rate limited", status_code=429),
        ),
    ):
        client.post("/ingest", json={"repo_url": "https://github.com/x/y"})
        r = client.post("/query", json={"question": "hi"})

    assert r.status_code == 429
