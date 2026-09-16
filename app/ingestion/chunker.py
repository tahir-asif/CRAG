import ast
from pathlib import Path

from app.config import MAX_FILE_SIZE_KB, SKIP_DIRS
from app.models import RetrievedChunk


def chunk_repo(repo_path: Path, extensions: list[str]) -> list[RetrievedChunk]:
    chunks = []
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue
        if file_path.suffix not in extensions:
            continue
        rel_path = str(file_path.relative_to(repo_path))
        if file_path.suffix == ".py":
            chunks.extend(chunk_python_file(file_path, rel_path))
        else:
            chunks.extend(
                _line_window_chunks(file_path.read_text(errors="ignore"), rel_path)
            )
    return chunks


def chunk_python_file(file_path: Path, rel_path: str) -> list[RetrievedChunk]:
    try:
        source = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    if len(source) / 1024 > MAX_FILE_SIZE_KB:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _line_window_chunks(source, rel_path)

    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = node.end_lineno or node.lineno
            lines = source.splitlines()[start - 1 : end]
            content = "\n".join(lines)
            chunks.append(
                RetrievedChunk(
                    content=content,
                    file_path=rel_path,
                    start_line=start,
                    end_line=end,
                    chunk_type="function"
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else "class",
                    name=node.name,
                    score=0.0,
                    source="ast",
                )
            )
    return chunks


def _line_window_chunks(
    source: str, rel_path: str, window: int = 50, overlap: int = 10
) -> list[RetrievedChunk]:
    lines = source.splitlines()
    chunks = []
    i = 0
    while i < len(lines):
        end = min(i + window, len(lines))
        content = "\n".join(lines[i:end])
        chunks.append(
            RetrievedChunk(
                content=content,
                file_path=rel_path,
                start_line=i + 1,
                end_line=end,
                chunk_type="module",
                name=None,
                score=0.0,
                source="fallback",
            )
        )
        if end == len(lines):
            break
        i += window - overlap
    return chunks
