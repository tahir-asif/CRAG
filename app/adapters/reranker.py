from typing import Protocol

from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL


class Reranker(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def build_reranker() -> Reranker:
    return CrossEncoderReranker()


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model = CrossEncoder(model_name, device="cpu")

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self._model.predict(pairs)]
