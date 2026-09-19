from app.ingestion.chunker import chunk_python_file, chunk_repo
from app.models import Chunk


def test_chunk_python_file_extracts_functions_and_classes(sample_repo_path):
    chunks = chunk_python_file(sample_repo_path / "auth.py", "auth.py")
    names = {c.name for c in chunks}
    assert names == {"authenticate", "TokenManager"}


def test_chunk_python_file_assigns_correct_types(sample_repo_path):
    chunks = chunk_python_file(sample_repo_path / "auth.py", "auth.py")
    types = {c.name: c.chunk_type for c in chunks}
    assert types["authenticate"] == "function"
    assert types["TokenManager"] == "class"


def test_chunk_python_file_line_ranges_are_valid(sample_repo_path):
    chunks = chunk_python_file(sample_repo_path / "auth.py", "auth.py")
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line
        assert c.content.strip()


def test_chunk_repo_walks_all_python_files(sample_repo_path):
    chunks = chunk_repo(sample_repo_path, [".py"])
    names = {c.name for c in chunks if c.name}
    assert "authenticate" in names
    assert "parse_token" in names


def test_chunk_repo_produces_chunk_not_retrieved_chunk(sample_repo_path):
    chunks = chunk_repo(sample_repo_path, [".py"])
    assert chunks
    for c in chunks:
        assert isinstance(c, Chunk)
        assert not hasattr(c, "score")
        assert not hasattr(c, "source")


def test_chunk_python_file_falls_back_on_syntax_error(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def unclosed(:\n    pass")
    chunks = chunk_python_file(bad, "bad.py")
    assert chunks
    assert all(c.chunk_type == "module" for c in chunks)


def test_chunk_repo_skips_ignored_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def real(): pass")
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "fake.py").write_text("def skipped(): pass")

    chunks = chunk_repo(tmp_path, [".py"])
    names = {c.name for c in chunks if c.name}
    assert "real" in names
    assert "skipped" not in names
