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

eval:  ## Run eval (REPO=<name>; VARIANT=<full,no-rerank,vector-only>; GEN=1 to run LLM)
	uv run python -m scripts.run_eval \
		--repo $(REPO) \
		--variant $(or $(VARIANT),full) \
		$(if $(GEN),--generate,)
