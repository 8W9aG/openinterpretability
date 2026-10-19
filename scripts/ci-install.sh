#!/usr/bin/env bash
# Install the locked dependencies with CPU-only torch, for CI.
#
# On Linux the locked torch pulls in several GB of CUDA wheels that tests never use.
# This installs every other package at its uv.lock version, and torch at the same
# version from the PyTorch CPU index.
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
REQS="$(mktemp)"
trap 'rm -f "$REQS"' EXIT

uv venv --quiet "$VENV_DIR"
uv export --frozen --no-hashes --group dev --no-emit-project --no-header > "$REQS"
TORCH="$(grep -E '^torch==' "$REQS" | cut -d' ' -f1)"
grep -vE '^(torch|triton|nvidia-|cuda-)' "$REQS" > "$REQS.filtered"
mv "$REQS.filtered" "$REQS"

PY="$VENV_DIR/bin/python"
uv pip install --quiet --python "$PY" "$TORCH" --index-url https://download.pytorch.org/whl/cpu
uv pip install --quiet --python "$PY" -r "$REQS"
uv pip install --quiet --python "$PY" --no-deps -e .
"$PY" -c "import torch, openinterp; print(f'torch {torch.__version__}, openinterp {openinterp.__version__}')"
