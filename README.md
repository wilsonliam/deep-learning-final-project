# Deep Learning Final Project

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you want the notebook stack too:

```bash
python -m pip install -e ".[dev,notebooks]"
```

## Common commands

Run the test suite:

```bash
python3 -m pytest
```

Run the pseudobulk CLI smoke test:

```bash
pseudobulk-tahoe100 --smoke
```

If you have not installed the console script, you can run it directly:

```bash
python3 scripts/pseudobulk_tahoe100.py --smoke
```

## Hugging Face on a compute box

Recommended auth setup:

```bash
export HF_TOKEN=your_hugging_face_token
export HF_XET_HIGH_PERFORMANCE=1
python3 scripts/pseudobulk_tahoe100.py --smoke
```

If your cluster has a faster scratch disk, point the Hugging Face cache there:

```bash
python3 scripts/pseudobulk_tahoe100.py --smoke --hf-cache-dir "${SCRATCH:-/tmp}/hf_cache"
```

Notes:

- The pseudobulk CLI targets the canonical Tahoe dataset id `tahoebio/Tahoe-100M`.
- `HF_TOKEN` is the preferred env var. `HUGGING_FACE_HUB_TOKEN` is still accepted for compatibility, but it is deprecated.
- If `HF_TOKEN` is not set, the script falls back to Hugging Face's normal auth resolution, which may use a saved `hf auth login` token or anonymous access.
- The warning about unauthenticated requests means lower rate limits and potentially slower downloads; it does not by itself mean the public Tahoe dataset load has failed.
- Set Hugging Face env vars in the shell before launching Python so the Hub client sees them during startup.
- `HF_HUB_ENABLE_HF_TRANSFER` is deprecated in current Hugging Face clients. Use `HF_XET_HIGH_PERFORMANCE=1` if your environment supports Xet-backed high-performance transfers.
- If startup stalls on metadata or downloads, `HF_HUB_ETAG_TIMEOUT` and `HF_HUB_DOWNLOAD_TIMEOUT` are the first timeout knobs to tune.
- The pseudobulk CLI now streams the live Tahoe `expression_data` schema (`drug`, `sample`, `plate`, `BARCODE_SUB_LIB_ID`, `cell_line_id`, `genes`, `expressions`), resolves `drugname_drugconc` through `sample_metadata`, and writes both `drug` and `drugname_drugconc` into each output `.h5ad`.
- Metadata integrity mismatches are fatal: missing `sample_metadata` rows, duplicate `sample` keys, or streamed `drug`/`plate` values that disagree with `sample_metadata` abort the run instead of silently regrouping rows.
