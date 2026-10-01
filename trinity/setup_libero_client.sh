#!/bin/bash
# One-time setup on PACE: builds the LIBERO simulator's own Python 3.8 environment.
# Run from the repo root inside a CPU job:  bash trinity/setup_libero_client.sh
set -euo pipefail
source ~/.openpi_env

git submodule update --init --depth 1   # LIBERO + ALOHA simulator assets, PACE only
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client -e third_party/libero
echo "LIBERO client environment ready."
