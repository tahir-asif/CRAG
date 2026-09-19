from unittest.mock import MagicMock

import groq
import pytest

from app.exceptions import LLMError
from app.generation import llm
from app.models import RetrievedChunk


def _chunk(name: str = "f") -> RetrievedChunk:
    return RetrievedChunk(
        content=f"def {name}(): pass",
        file_path=f"{name}.py",
        start_line=10,
        end_line=12,
        chunk_type="function",
        name=name,
        score=0.5,
        source="vector",
    )


def _groq_error(cls, **attrs):
    """Construct a Groq exception without running its __init__.

    The SDK's exception constructors expect httpx.Response objects we
    don't want to build. __new__ gives us an instance that passes
    isinstance checks without touching the real constructor.
    """
    exc = cls.__new__(cls)
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


def _raising_client(exc):
    client = MagicMock()
    client.chat.completions.create.side_effect = exc
    return client


# --- Prompt building --------------------------------------------------------


def test_build_prompt_includes_chunk_metadata_and_question():
    chunks = [_chunk("alpha"), _chunk("beta")]
    prompt = llm._build_prompt("How does alpha work?", chunks)

    assert "alpha.py" in prompt
    assert "beta.py" in prompt
    assert "How does alpha work?" in prompt
    assert "def alpha(): pass" in prompt


# --- Error translation ------------------------------------------------------


def test_auth_error_with_user_key_returns_401():
    client = _raising_client(_groq_error(groq.AuthenticationError))
    with pytest.raises(LLMError) as info:
        llm._get_response(client, "p", api_key="user-key")
    assert info.value.status_code == 401


def test_auth_error_with_server_key_returns_500():
    client = _raising_client(_groq_error(groq.AuthenticationError))
    with pytest.raises(LLMError) as info:
        llm._get_response(client, "p", api_key=None)
    assert info.value.status_code == 500


def test_rate_limit_returns_429():
    client = _raising_client(_groq_error(groq.RateLimitError))
    with pytest.raises(LLMError) as info:
        llm._get_response(client, "p")
    assert info.value.status_code == 429


def test_timeout_returns_504():
    client = _raising_client(_groq_error(groq.APITimeoutError))
    with pytest.raises(LLMError) as info:
        llm._get_response(client, "p")
    assert info.value.status_code == 504


def test_connection_error_returns_503():
    client = _raising_client(_groq_error(groq.APIConnectionError))
    with pytest.raises(LLMError) as info:
        llm._get_response(client, "p")
    assert info.value.status_code == 503


def test_api_status_error_returns_502_and_includes_code():
    exc = _groq_error(groq.APIStatusError, status_code=418)
    client = _raising_client(exc)
    with pytest.raises(LLMError) as info:
        llm._get_response(client, "p")
    assert info.value.status_code == 502
    assert "418" in str(info.value)


# --- generate_answer edge cases ---------------------------------------------


def test_missing_api_key_raises_500(monkeypatch):
    monkeypatch.setattr(llm, "GROQ_API_KEY", None)
    with pytest.raises(LLMError) as info:
        llm.generate_answer("q", [], api_key=None)
    assert info.value.status_code == 500


def test_empty_content_raises_502(monkeypatch):
    monkeypatch.setattr(llm, "GROQ_API_KEY", "fake-key")

    response = MagicMock()
    response.choices[0].message.content = None
    monkeypatch.setattr(llm, "_get_response", lambda *a, **k: response)

    with pytest.raises(LLMError) as info:
        llm.generate_answer("q", [])
    assert info.value.status_code == 502
