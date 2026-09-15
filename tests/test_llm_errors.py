from unittest.mock import MagicMock

import groq
import pytest

from app.exceptions import LLMError
from app.generation import llm


def _fake_client_raising(exc):
    client = MagicMock()
    client.chat.completions.create.side_effect = exc
    return client


def test_auth_error_with_user_key_returns_401(monkeypatch):
    monkeypatch.setattr(
        llm,
        "Groq",
        lambda api_key: _fake_client_raising(
            groq.AuthenticationError("bad", response=MagicMock(), body=None)
        ),
    )
    with pytest.raises(LLMError) as exc_info:
        llm.generate_answer("q", [], api_key="user-key")
    assert exc_info.value.status_code == 401


def test_auth_error_with_server_key_returns_500(monkeypatch):
    monkeypatch.setattr(llm, "GROQ_API_KEY", "server-key")
    monkeypatch.setattr(
        llm,
        "Groq",
        lambda api_key: _fake_client_raising(
            groq.AuthenticationError("bad", response=MagicMock(), body=None)
        ),
    )
    with pytest.raises(LLMError) as exc_info:
        llm.generate_answer("q", [])
    assert exc_info.value.status_code == 500


def test_missing_key_raises_500(monkeypatch):
    monkeypatch.setattr(llm, "GROQ_API_KEY", None)
    with pytest.raises(LLMError) as exc_info:
        llm.generate_answer("q", [], api_key=None)
    assert exc_info.value.status_code == 500
