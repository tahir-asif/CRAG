from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_repo
from app.ingestion.clone import clone_repo


@pytest.mark.live
def test_clone_and_chunk_real_repo(tmp_path, monkeypatch):
    """Clone a real repo via GitHub and confirm we get chunks.

    Verifies GitPython works, the branch-default logic handles `master`
    (requests-html uses it), and the chunker handles real code.
    """
    monkeypatch.setattr("app.ingestion.clone.REPO_CACHE_DIR", str(tmp_path))
    # Re-import to pick up the patched constant in clone.py's module scope
    import importlib

    from app.ingestion import clone as clone_module

    importlib.reload(clone_module)

    repo_path: Path = clone_module.clone_repo(
        "https://github.com/psf/requests-html", branch=None
    )
    assert repo_path.exists()

    chunks = chunk_repo(repo_path, [".py"])
    assert chunks, "no chunks produced from real repo"
    assert any(c.chunk_type == "function" for c in chunks)
