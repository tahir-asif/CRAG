import argparse
from unittest.mock import MagicMock, patch

import pytest
import requests

from cli import _request, cmd_ingest, cmd_query, cmd_repos

# --- helpers ---------------------------------------------------------------


def _args(**kwargs) -> argparse.Namespace:
    defaults = {"base_url": "http://testserver"}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _ok(body: dict) -> MagicMock:
    r = MagicMock()
    r.ok = True
    r.status_code = 200
    r.json.return_value = body
    return r


def _err(status: int, *, json_body: dict | None = None, text: str = "") -> MagicMock:
    r = MagicMock()
    r.ok = False
    r.status_code = status
    r.text = text
    if json_body is not None:
        r.json.return_value = json_body
    else:
        r.json.side_effect = ValueError("not JSON")
    return r


# --- cmd_repos -------------------------------------------------------------


def test_cmd_repos_prints_each_repo(capsys):
    body = {"repos": ["click", "requests-html"]}
    with patch("cli.requests.request", return_value=_ok(body)) as mock_req:
        cmd_repos(_args())

    mock_req.assert_called_once_with("GET", "http://testserver/repos")
    out = capsys.readouterr().out
    assert "click" in out
    assert "requests-html" in out


# --- cmd_ingest ------------------------------------------------------------


def test_cmd_ingest_sends_url_and_extensions(capsys):
    body = {"repo_name": "click", "files_indexed": 42, "chunks_created": 800}
    with patch("cli.requests.request", return_value=_ok(body)) as mock_req:
        cmd_ingest(
            _args(repo_url="https://github.com/pallets/click", branch=None, ext=[".py"])
        )

    call = mock_req.call_args
    assert call.args == ("POST", "http://testserver/ingest")
    assert call.kwargs["json"] == {
        "repo_url": "https://github.com/pallets/click",
        "file_extensions": [".py"],
    }

    out = capsys.readouterr().out
    assert "click" in out
    assert "42" in out
    assert "800" in out


def test_cmd_ingest_includes_branch_when_set():
    body = {"repo_name": "click", "files_indexed": 1, "chunks_created": 1}
    with patch("cli.requests.request", return_value=_ok(body)) as mock_req:
        cmd_ingest(
            _args(repo_url="https://github.com/x/y", branch="develop", ext=[".py"])
        )

    assert mock_req.call_args.kwargs["json"]["branch"] == "develop"


def test_cmd_ingest_omits_branch_when_none():
    body = {"repo_name": "click", "files_indexed": 1, "chunks_created": 1}
    with patch("cli.requests.request", return_value=_ok(body)) as mock_req:
        cmd_ingest(_args(repo_url="https://github.com/x/y", branch=None, ext=[".py"]))

    assert "branch" not in mock_req.call_args.kwargs["json"]


# --- cmd_query -------------------------------------------------------------


def test_cmd_query_prints_answer_and_citations(capsys):
    body = {
        "answer": "The echo function writes to stdout.",
        "citations": [
            {"file_path": "utils.py", "start_line": 10, "end_line": 20, "name": "echo"},
        ],
        "retrieved_chunks": [],
    }
    with patch("cli.requests.request", return_value=_ok(body)) as mock_req:
        cmd_query(_args(question="How does echo work?", repo="click"))

    sent = mock_req.call_args.kwargs["json"]
    assert sent == {"question": "How does echo work?", "repo_name": "click"}

    out = capsys.readouterr().out
    assert "The echo function writes to stdout." in out
    assert "utils.py" in out
    assert "echo" in out


# --- _request --------------------------------------------------------------


def test_request_returns_json_on_success():
    with patch("cli.requests.request", return_value=_ok({"hello": "world"})):
        assert _request("http://x", "GET", "/y") == {"hello": "world"}


def test_request_connection_error_exits(capsys):
    with (
        patch("cli.requests.request", side_effect=requests.ConnectionError("refused")),
        pytest.raises(SystemExit) as info,
    ):
        _request("http://x", "GET", "/y")

    assert info.value.code == 1
    assert "Connection error" in capsys.readouterr().err


def test_request_http_error_with_json_detail(capsys):
    err = _err(404, json_body={"detail": "Repo 'nope' not found."})
    with (
        patch("cli.requests.request", return_value=err),
        pytest.raises(SystemExit) as info,
    ):
        _request("http://x", "GET", "/y")

    assert info.value.code == 1
    err_out = capsys.readouterr().err
    assert "Error (404)" in err_out
    assert "Repo 'nope' not found." in err_out


def test_request_http_error_with_non_json_body(capsys):
    err = _err(500, text="<html>Internal Server Error</html>")
    with (
        patch("cli.requests.request", return_value=err),
        pytest.raises(SystemExit),
    ):
        _request("http://x", "GET", "/y")

    assert "Internal Server Error" in capsys.readouterr().err
