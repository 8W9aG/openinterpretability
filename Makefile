.PHONY: install test lint format check stats build search-index sample-run serve loop clean

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

# Experiments, units, models, VERIFIED and self-reviewed shares, and protocol versions.
stats:
	uv run openinterp stats --wiki-dir $(WIKI_DIR)

format:
	uv run ruff format .
	uv run ruff check --fix .

build:
	uv run openinterp build --wiki-dir $(WIKI_DIR) --output-dir $(DIST_DIR)
	@$(MAKE) --no-print-directory search-index

# Pagefind (a locked dev dependency, so no Node is needed) indexes dist/ for search.
search-index:
	uv run python -m pagefind --site $(DIST_DIR)

# A few real experiments from the wiki covering each verdict, a neuron with several
# experiments, and held-out explanation scores. Records are write-once, so paths are stable.
SAMPLE_RECORDS = $(addprefix wiki/experiments/Qwen--Qwen2.5-0.5B/, \
	L4/N1889-positive/20261019T113221Z.json \
	L9/N1458-negative/20261020T211015Z.json \
	L11/N4255-positive/20261019T050928Z.json \
	L11/N4255-positive/20261020T171549Z.json \
	L12/N4232-negative/20261021T044854Z.json \
	L20/N967-negative/20261021T040406Z.json)

# Builds the site from a small sample of real records (no model or LLM calls).
sample-run:
	rm -rf build/sample-records build/sample-wiki
	mkdir -p build/sample-records
	cp $(SAMPLE_RECORDS) build/sample-records/
	uv run openinterp wiki --import-dir build/sample-records --wiki-dir build/sample-wiki
	@$(MAKE) --no-print-directory build WIKI_DIR=build/sample-wiki
	@echo "Open $(DIST_DIR)/index.html, or run 'make serve' for DuckDB/Pagefind (needs HTTP)."

serve:
	uv run python -m http.server --directory $(DIST_DIR) 8000

# Run experiments back to back with headless Claude Code; see scripts/experiment-loop.sh --help.
loop:
	scripts/experiment-loop.sh $(LOOP_ARGS)

clean:
	rm -rf $(DIST_DIR) .pytest_cache .ruff_cache
