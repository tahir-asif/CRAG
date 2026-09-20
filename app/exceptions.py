class DomainError(Exception):
    """Base for all application domain errors.

    Every subclass defines default_message and default_status_code.
    Any raise site can override either or both.
    """

    default_message: str = "An unexpected error occurred."
    default_status_code: int = 500

    def __init__(self, message: str | None = None, status_code: int | None = None):
        self.message = message if message is not None else self.default_message
        self.status_code = (
            status_code if status_code is not None else self.default_status_code
        )
        super().__init__(self.message)


class IngestionError(DomainError):
    default_message = "Ingestion failed."
    default_status_code = 500


class RetrievalError(DomainError):
    default_message = "Retrieval failed."
    default_status_code = 500


class LLMError(DomainError):
    default_message = "LLM generation failed."
    default_status_code = 502


class NoReposError(DomainError):
    default_message = "No repos ingested. Call /ingest first."
    default_status_code = 400


class RepoNotFoundError(DomainError):
    default_message = "Repo not found."
    default_status_code = 404


class AmbiguousRepoError(DomainError):
    default_message = "Multiple repos indexed. Specify a repo name."
    default_status_code = 400
