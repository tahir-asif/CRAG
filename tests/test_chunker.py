from pathlib import Path

from app.ingestion.chunker import chunk_python_file


def test_chunks_functions_and_classes():
    path = Path("tests/fixtures/sample_repo/sample.py")
    chunks = chunk_python_file(path, "sample.py")
    names = {c.name for c in chunks}
    assert names == {"hello", "Greeter"}
    hello = next(c for c in chunks if c.name == "hello")
    assert hello.start_line == 1
    assert hello.end_line == 2
    assert hello.chunk_type == "function"
