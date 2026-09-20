from app.models import RetrievedChunk


def hit_at_k(chunks: list[RetrievedChunk], expected_files: list[str], k: int) -> float:
    """1.0 if any expected file appears in the top-k chunks, else 0.0."""
    top_files = {c.file_path for c in chunks[:k]}
    return 1.0 if any(f in top_files for f in expected_files) else 0.0


def reciprocal_rank(chunks: list[RetrievedChunk], expected_files: list[str]) -> float:
    """1/rank of the first relevant chunk; 0.0 if none found."""
    for i, c in enumerate(chunks, start=1):
        if c.file_path in expected_files:
            return 1.0 / i
    return 0.0


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Fraction of keywords present in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return matched / len(keywords)
