# This project's Make targets. The generic ones are in labinacan.mk, which reads the values
# below (#6630). The targets here are this project's own.
DIST_NAME := open-interpretability
PACKAGES := openinterpretability
# The heavy lock keeps two heavy jobs, such as a test run and a timing, from overlapping.
TEST_WRAPPER := python -m openinterpretability.core.heavy_lock
CSS_MODULE := openinterpretability.builder.css
BUILD_COMMAND = $(BUILD_WRAPPER) uv run open-interpretability build --wiki-dir $(WIKI_DIR) \
	--output-dir $(DIST_DIR)
CACHE_DIR_COMMAND = uv run python -c \
	"from openinterpretability.core.legacy_names import cache_base; print(cache_base())"
PROJECT_LINT := project-lint

include labinacan.mk

.PHONY: project-lint perf check check-all stats maps sample-run loop

# This project's checks inside `make lint`: the data files, and the old package name.
project-lint:
	uv run python scripts/check_data_files.py
	python3 scripts/check_old_name.py

# Wall time, CPU time and peak memory of fixed workloads, checked against
# docs/analyses/perf-baseline.json (scripts/perf_bench.py). Not in CI: runner timings vary.
# The heavy lock keeps another heavy job from overlapping the timings.
perf:
	uv run python -m openinterpretability.core.heavy_lock python scripts/perf_bench.py

# What a pull request's check runs: only the branch's changes, unless it changes code.
BASE_REF ?= origin/main
check:
	uv run open-interpretability check --base-ref $(BASE_REF)

# What CI runs on main: every record.
check-all:
	uv run open-interpretability check

# Experiments, units, models, VERIFIED and self-reviewed shares, and protocol versions.
stats:
	uv run open-interpretability stats --wiki-dir $(WIKI_DIR)

# The shared concept map of every configured model, refreshed by each release (#4785, #4586).
# Old poles keep their places. A refit (`open-interpretability map --refit`) is a separate PR.
maps:
	uv run --extra map open-interpretability map --wiki-dir $(WIKI_DIR)

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
	uv run open-interpretability wiki --import-dir build/sample-records --wiki-dir build/sample-wiki
	@$(MAKE) --no-print-directory build WIKI_DIR=build/sample-wiki
	@echo "Open $(DIST_DIR)/index.html, or run 'make serve' for DuckDB/Pagefind (needs HTTP)."

# Run experiments back to back with headless Claude Code; see scripts/experiment-loop.sh --help.
loop:
	scripts/experiment-loop.sh $(LOOP_ARGS)
