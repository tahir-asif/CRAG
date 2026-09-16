import shutil
from pathlib import Path

import git

from app.config import REPO_CACHE_DIR


class IngestionError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def clone_repo(repo_url: str, branch: str = "main") -> Path:
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = Path(REPO_CACHE_DIR) / repo_name

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        git.Repo.clone_from(repo_url, dest, branch=branch, depth=1)
    except git.GitCommandError as e:
        raise IngestionError(f"Failed to clone {repo_url}: {e}", status_code=400)
    except git.NoSuchPathError as e:
        raise IngestionError(f"Invalid destination path: {e}", status_code=500)

    return dest
