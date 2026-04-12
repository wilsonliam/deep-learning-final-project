#!/usr/bin/env bash
set -euo pipefail

# ---- user-configurable settings ----
export HF_TOKEN=""
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=120
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

REPO_DIR="/deep-learning-final-project"
PYTHON_BIN="python3"
SESSION_NAME="pseudobulk_full"
OUTPUT_DIR="/pseudobulk_output"
CACHE_DIR="${SCRATCH:-/tmp}/hf_cache"

# Optional extra flags:
# EXTRA_ARGS="--cell-lines CVCL_0480 CVCL_C466"
EXTRA_ARGS=""

# ---- checks ----
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required so the run survives disconnects while keeping the progress bar visible."
  echo "Install it first, for example: sudo apt-get update && sudo apt-get install -y tmux"
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

CMD=$(cat <<EOF
cd "${REPO_DIR}"
exec ${PYTHON_BIN} scripts/pseudobulk_tahoe100.py \
  --hf-cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --progress bar \
  --progress-every 10000 \
  ${EXTRA_ARGS}
EOF
)

tmux new-session -d -s "${SESSION_NAME}" "bash -lc '${CMD}'"

echo "Started detached tmux session: ${SESSION_NAME}"
echo "Attach to watch progress: tmux attach -t ${SESSION_NAME}"
echo "Detach without stopping: Ctrl-b then d"
echo "Output directory: ${OUTPUT_DIR}"

