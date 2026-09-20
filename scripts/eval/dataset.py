import json
from pathlib import Path

from pydantic import BaseModel


class QAPair(BaseModel):
    question: str
    expected_files: list[str]
    expected_keywords: list[str] = []


def load_qa_pairs(path: str | Path) -> list[QAPair]:
    data = json.loads(Path(path).read_text())
    return [QAPair.model_validate(item) for item in data]
