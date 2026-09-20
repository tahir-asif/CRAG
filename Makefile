.PHONY: help dev test test-live eval install

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with uv
	uv sync

dev:  ## Run dev server with auto-reload (watches app/ only)
	uv run uvicorn app.main:app --reload --reload-dir app --no-access-log

test:  ## Run the offline test suite
	uv run pytest -v

test-live:  ## Run all tests including live ones (uses Groq tokens)
	uv run pytest --live -v

eval:  ## Run the eval harness (REPO=click; add GEN=1 to also call the LLM)
	uv run python -m scripts.run_eval --repo $(REPO) $(if $(GEN),--generate,)
