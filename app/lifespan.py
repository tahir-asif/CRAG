import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.dependencies import build_default_dependencies

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building dependencies...")
    app.state.deps = build_default_dependencies()
    logger.info("Dependencies ready.")
    yield
    app.state.deps = None
    logger.info("Dependencies released.")
