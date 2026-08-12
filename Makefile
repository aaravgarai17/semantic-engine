# Common tasks. `make help` lists them.
.DEFAULT_GOAL := help
.PHONY: help install test cov verify clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv and install dependencies
	python3 -m venv .venv
	./.venv/bin/pip install -q -r requirements.txt
	@echo "done — activate with: source .venv/bin/activate"

test:  ## Run the test suite
	pytest -q

cov:  ## Run tests with a coverage report
	pytest --cov=app --cov-report=term-missing -q

verify:  ## Run the full verification script
	./verify.sh

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage .coverage.* htmlcov

.PHONY: run eval eval-fast sweep
run:  ## Start the API
	uvicorn app.main:app --reload

eval:  ## Retrieval evaluation with real embeddings (needs sentence-transformers)
	python -m eval.run_eval --embedder local --examples

eval-fast:  ## Retrieval evaluation offline, non-semantic embedder
	python -m eval.run_eval --embedder hash

sweep:  ## Chunk-size and fusion-weight experiments
	python -m eval.run_eval --embedder local --sweep
	python -m eval.run_eval --embedder local --weight-sweep
