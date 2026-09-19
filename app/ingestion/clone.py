import logging
import shutil
from pathlib import Path

import git

from app.config import DEFAULT_INGEST_DEPTH, REPO_CACHE_DIR
from app.exceptions import IngestionError

logger = logging.getLogger(__name__)


def clone_repo(repo_url: str, branch: str | None = None) -> Path:
    logger.info("Cloning %s (branch=%s)", repo_url, branch or "default")
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = Path(REPO_CACHE_DIR) / repo_name

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        if branch is None:
            git.Repo.clone_from(repo_url, dest, depth=DEFAULT_INGEST_DEPTH)
        else:
            git.Repo.clone_from(
                repo_url, dest, branch=branch, depth=DEFAULT_INGEST_DEPTH
            )
        logger.info("Cloned %s to %s", repo_url, dest)

    except git.GitCommandError as e:
        if branch is not None and "Remote branch" in str(e):
            raise IngestionError(
                f"Branch '{branch}' not found in {repo_url}.",
                status_code=404,
            )
        raise IngestionError(f"Failed to clone {repo_url}: {e}", status_code=400)

    except git.NoSuchPathError as e:
        raise IngestionError(f"Invalid destination path: {e}", status_code=500)

    return dest
