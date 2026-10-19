.PHONY: install test lint format check build search-index sample-run serve clean

AUDITS_DIR ?= audits
DIST_DIR ?= dist

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run python scripts/check_line_limit.py

check:
	uv run openinterp check

format:
	uv run ruff format .
	uv run ruff check --fix .

build:
	uv run openinterp build --audits-dir $(AUDITS_DIR) --output-dir $(DIST_DIR)
	@$(MAKE) --no-print-directory search-index

# Pagefind (a locked dev dependency, so no Node is needed) indexes dist/ for search.
search-index:
	uv run python -m pagefind --site $(DIST_DIR)

# Builds the site from synthetic example records (no model or LLM calls).
sample-run:
	uv run python examples/make_sample_audits.py
	@$(MAKE) --no-print-directory build AUDITS_DIR=examples/audits
	@echo "Open $(DIST_DIR)/index.html, or run 'make serve' for DuckDB/Pagefind (needs HTTP)."

serve:
	uv run python -m http.server --directory $(DIST_DIR) 8000

clean:
	rm -rf $(DIST_DIR) .pytest_cache .ruff_cache
