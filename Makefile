.PHONY: install test lint format check build search-index sample-run serve loop clean

WIKI_DIR ?= wiki
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
	uv run openinterp build --wiki-dir $(WIKI_DIR) --output-dir $(DIST_DIR)
	@$(MAKE) --no-print-directory search-index

# Pagefind (a locked dev dependency, so no Node is needed) indexes dist/ for search.
search-index:
	uv run python -m pagefind --site $(DIST_DIR)

# Builds the site from synthetic example records (no model or LLM calls).
sample-run:
	uv run python examples/make_sample_audits.py
	rm -rf build/sample-wiki
	uv run openinterp wiki --import-dir examples/audits --wiki-dir build/sample-wiki
	@$(MAKE) --no-print-directory build WIKI_DIR=build/sample-wiki
	@echo "Open $(DIST_DIR)/index.html, or run 'make serve' for DuckDB/Pagefind (needs HTTP)."

serve:
	uv run python -m http.server --directory $(DIST_DIR) 8000

# Run experiments back to back with headless Claude Code; see scripts/experiment-loop.sh --help.
loop:
	scripts/experiment-loop.sh $(LOOP_ARGS)

clean:
	rm -rf $(DIST_DIR) .pytest_cache .ruff_cache
