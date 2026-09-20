from app.eval.metrics import hit_at_k, keyword_coverage, reciprocal_rank
from app.models import RetrievedChunk


def _chunk(file_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        content="x",
        file_path=file_path,
        start_line=1,
        end_line=1,
        chunk_type="function",
        name=None,
        score=1.0,
        source="rerank",
    )


def test_hit_at_k_finds_expected_file():
    chunks = [_chunk("a.py"), _chunk("b.py"), _chunk("c.py")]
    assert hit_at_k(chunks, ["b.py"], k=5) == 1.0
    assert hit_at_k(chunks, ["z.py"], k=5) == 0.0


def test_hit_at_k_respects_cutoff():
    chunks = [_chunk("a.py"), _chunk("b.py"), _chunk("c.py")]
    assert hit_at_k(chunks, ["c.py"], k=2) == 0.0
    assert hit_at_k(chunks, ["c.py"], k=3) == 1.0


def test_reciprocal_rank_returns_inverse_position():
    chunks = [_chunk("a.py"), _chunk("b.py")]
    assert reciprocal_rank(chunks, ["a.py"]) == 1.0
    assert reciprocal_rank(chunks, ["b.py"]) == 0.5
    assert reciprocal_rank(chunks, ["z.py"]) == 0.0


def test_keyword_coverage_case_insensitive():
    assert keyword_coverage("The ECHO function", ["echo"]) == 1.0
    assert keyword_coverage("nothing relevant", ["echo"]) == 0.0
    assert keyword_coverage("echo handles output", ["echo", "nl"]) == 0.5


def test_keyword_coverage_empty_keywords_returns_one():
    assert keyword_coverage("anything", []) == 1.0
