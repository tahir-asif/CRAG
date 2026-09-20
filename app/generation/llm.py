import json
import logging

import groq
from groq import Groq
from groq.types.chat import ChatCompletion

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.exceptions import LLMError
from app.models import LLMResponse, RetrievedChunk

logger = logging.getLogger(__name__)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    api_key: str | None = None,
) -> LLMResponse:
    key = api_key or GROQ_API_KEY
    if not key:
        raise LLMError("Server is missing an LLM API key.", status_code=500)

    client = Groq(api_key=key)
    prompt = _build_prompt(question, chunks)
    response = _get_response(client, prompt, api_key=api_key)

    content = response.choices[0].message.content
    if content is None:
        raise LLMError("LLM returned an empty response.", status_code=502)

    try:
        parsed = json.loads(content)
        answer = str(parsed.get("answer", "")).strip()
        if not answer:
            raise ValueError("empty answer field")
        raw_indices = parsed.get("cited_indices", [])
        cited_indices = [int(i) for i in raw_indices if isinstance(i, (int, float))]
        return LLMResponse(answer=answer, cited_indices=cited_indices)
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        logger.warning("LLM did not return parseable JSON; using raw content")
        return LLMResponse(answer=content, cited_indices=[])


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    numbered = []
    for i, c in enumerate(chunks):
        numbered.append(
            f"[{i + 1}] {c.file_path}:{c.start_line}-{c.end_line}\n{c.content}"
        )
    context = "\n\n".join(numbered)

    return (
        "Answer the question using ONLY the code snippets below.\n"
        "Return a JSON object with two keys:\n"
        '  "answer": your answer as a string\n'
        '  "cited_indices": list of snippet numbers you actually used\n\n'
        f"Snippets:\n{context}\n\n"
        f"Question: {question}\n"
        "Respond with JSON only:"
    )


def _get_response(
    client: Groq, prompt: str, api_key: str | None = None
) -> ChatCompletion:
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500,
        )
        return response
    except groq.AuthenticationError:
        # If the caller passed their own key, it's their fault (401).
        # Otherwise the server's key is bad (500).
        raise LLMError("Invalid Groq API key.", status_code=401 if api_key else 500)
    except groq.RateLimitError:
        raise LLMError("LLM rate limit reached. Try again shortly.", status_code=429)
    except groq.APITimeoutError:
        raise LLMError("LLM request timed out.", status_code=504)
    except groq.APIConnectionError:
        raise LLMError("Could not reach the LLM provider.", status_code=503)
    except groq.APIStatusError as e:
        raise LLMError(f"LLM provider error ({e.status_code}).", status_code=502)
    except groq.APIError:
        raise LLMError("Unexpected LLM error.", status_code=502)
