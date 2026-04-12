python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export HF_TOKEN='your_hf_token_here'

export HF_CACHE_DIR="${SCRATCH:-/tmp}/hf_cache"
mkdir -p "$HF_CACHE_DIR"

nohup python scripts/pseudobulk_tahoe100.py \
  --sample-size 2000 \
  --block-size 20 \
  --num-workers 4 \
  --output-dir data/pseudobulk_smoke \
  --hf-cache-dir "$HF_CACHE_DIR" \
  > pseudobulk_smoke.log 2>&1 &

