import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions import DomainError

logger = logging.getLogger(__name__)


async def domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, DomainError):
        logger.error("%s: %s", type(exc).__name__, exc, exc_info=exc)
        raise TypeError(
            f"domain_error_handler received {type(exc).__name__}, expected DomainError"
        )

    if exc.status_code >= 500:
        logger.error("%s: %s", type(exc).__name__, exc, exc_info=exc)
    else:
        logger.warning("%s: %s", type(exc).__name__, exc)

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc)},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, domain_error_handler)
