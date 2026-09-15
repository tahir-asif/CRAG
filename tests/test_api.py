from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_query_without_ingest_returns_400():
    # reset stub store between tests — see note below
    from app import main

    main._STUB_STORE.clear()
    r = client.post("/query", json={"question": "hi"})
    assert r.status_code == 400


def test_query_with_bad_repo_returns_404():
    client.post("/ingest", json={"repo_url": "https://github.com/x/y"})
    r = client.post("/query", json={"question": "hi", "repo_name": "nope"})
    assert r.status_code == 404


@patch("app.main.generate_answer", return_value="mocked answer")
def test_query_success(mock_gen):
    client.post("/ingest", json={"repo_url": "https://github.com/x/y"})
    r = client.post("/query", json={"question": "hi"})
    assert r.status_code == 200
    assert r.json()["answer"] == "mocked answer"
