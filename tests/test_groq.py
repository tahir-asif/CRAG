import os

import pytest
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)
def test_groq_connection():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Reply with exactly: connection works"}],
        temperature=0,
        max_tokens=20,
    )
    content = response.choices[0].message.content
    assert content is not None
    assert "connection works" in content.lower()
