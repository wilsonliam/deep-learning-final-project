cd /Users/liamwilson/deep-learning-final-project

rm -rf .venv
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate

python --version
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export HF_TOKEN='your_hf_token_here'
export HF_CACHE_DIR="${SCRATCH:-/tmp}/hf_cache"
mkdir -p "$HF_CACHE_DIR"

python scripts/pseudobulk_tahoe100.py \
  --sample-size 2000 \
  --block-size 20 \
  --num-workers 4 \
  --output-dir data/pseudobulk_smoke \
  --hf-cache-dir "$HF_CACHE_DIR"

