#!/usr/bin/env python3
"""Pseudobulk Tahoe-100M single-cell data into per-cell-line AnnData files.

For each replicate-aware group
``(cell_line, drug+concentration, sample, plate, BARCODE_SUB_LIB_ID)``, cells
are partitioned into non-overlapping blocks of exactly ``block_size`` (default
200). Each block is summed, then the block-sums are averaged. Cells beyond the
last complete block are dropped. The output is one ``.h5ad`` file per cell
line containing an ``n_groups × n_genes`` matrix, with replicate metadata
stored in ``adata.obs`` for downstream batch-effect modeling and replicate QC.

Parallelism
-----------
HuggingFace streaming reads happen on the main process.  Per-record
densification can be parallelised across ``--num-workers`` multiprocessing
workers, but results are yielded back in source order so the accumulator sees
the same cell order as a serial run.

Memory
------
Accumulator memory ≈ 2 × n_groups × n_genes × 4 bytes (~500 KB per group).
At ``--sample-size 200000`` the working set is well under 2 GB.  A full 100M
run would likely cross 25 GB — a per-cell-line re-streaming refactor would
be needed for that.

Dependencies: datasets, anndata, numpy, pandas, torch, scipy
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

DATASET_NAME = "vevotx/Tahoe-100M"
HF_TOKEN_ENV_VAR = "HF_TOKEN"
HF_TOKEN_ENV_VAR_LEGACY = "HUGGING_FACE_HUB_TOKEN"
GROUPBY_OBS_COLUMNS = (
    "drugname_drugconc",
    "sample",
    "plate",
    "BARCODE_SUB_LIB_ID",
)
OBS_INDEX_NAME = "pseudobulk_group"


@dataclass(frozen=True, order=True)
class PseudobulkGroupKey:
    """Replicate-aware aggregation key for one pseudobulk row."""

    cell_line_id: str
    drug_key: str
    sample: str = ""
    plate: str = ""
    barcode_sub_lib_id: str = ""

    @property
    def replicate_id(self) -> str:
        """Return the best available replicate identifier for display."""
        return self.barcode_sub_lib_id or self.sample or self.plate or self.drug_key

    @property
    def obs_index(self) -> str:
        """Stable obs index string for this replicate-aware group."""
        return (
            f"drug={self.drug_key}|sample={self.sample}|plate={self.plate}"
            f"|barcode={self.barcode_sub_lib_id}"
        )

# ---------------------------------------------------------------------------
# BlockAccumulator
# ---------------------------------------------------------------------------

@dataclass
class BlockAccumulator:
    """Streaming block-sum-then-average accumulator for one pseudobulk group.

    Cells are fed one at a time via :meth:`add_cell`.  Every ``block_size``
    cells the running block sum is folded into ``cumulative_block_sum`` and the
    running buffer is reset.  At :meth:`finalize` time any incomplete trailing
    block is discarded.
    """

    n_genes: int
    block_size: int
    running_block_sum: np.ndarray = field(init=False, repr=False)
    cells_in_running_block: int = field(init=False, default=0)
    cumulative_block_sum: np.ndarray = field(init=False, repr=False)
    n_complete_blocks: int = field(init=False, default=0)
    n_cells_seen: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        assert self.block_size > 0, f"block_size must be > 0, got {self.block_size}"
        assert self.n_genes > 0, f"n_genes must be > 0, got {self.n_genes}"
        self.running_block_sum = np.zeros(self.n_genes, dtype=np.float64)
        self.cumulative_block_sum = np.zeros(self.n_genes, dtype=np.float64)

    def add_cell(self, expr: np.ndarray) -> None:
        """Add a single cell's dense expression vector."""
        assert expr.shape == (self.n_genes,), (
            f"Expected shape ({self.n_genes},), got {expr.shape}"
        )
        self.running_block_sum += expr
        self.cells_in_running_block += 1
        self.n_cells_seen += 1
        if self.cells_in_running_block == self.block_size:
            self.cumulative_block_sum += self.running_block_sum
            self.n_complete_blocks += 1
            self.running_block_sum.fill(0.0)
            self.cells_in_running_block = 0

    def finalize(self) -> np.ndarray | None:
        """Return the pseudobulk vector, or ``None`` if no complete blocks."""
        if self.n_complete_blocks == 0:
            return None
        return (self.cumulative_block_sum / self.n_complete_blocks).astype(
            np.float32
        )

    @property
    def n_cells_used(self) -> int:
        return self.block_size * self.n_complete_blocks

    @property
    def n_cells_dropped(self) -> int:
        return self.n_cells_seen - self.n_cells_used


# ---------------------------------------------------------------------------
# Gene vocabulary
# ---------------------------------------------------------------------------

def _resolve_hf_load_kwargs(
    hf_cache_dir: Path | None = None,
) -> tuple[dict[str, str], str | None]:
    """Build shared ``load_dataset`` kwargs from env auth and optional cache."""
    token = os.getenv(HF_TOKEN_ENV_VAR)
    token_source = None
    if token:
        token_source = HF_TOKEN_ENV_VAR
    else:
        legacy_token = os.getenv(HF_TOKEN_ENV_VAR_LEGACY)
        if legacy_token:
            token = legacy_token
            token_source = HF_TOKEN_ENV_VAR_LEGACY
            log.warning(
                "Environment variable %s is deprecated; prefer %s.",
                HF_TOKEN_ENV_VAR_LEGACY,
                HF_TOKEN_ENV_VAR,
            )

    load_kwargs: dict[str, str] = {}
    if token is not None:
        load_kwargs["token"] = token
    if hf_cache_dir is not None:
        load_kwargs["cache_dir"] = str(hf_cache_dir)
    return load_kwargs, token_source


def _log_hf_startup_config(
    token_source: str | None,
    hf_cache_dir: Path | None,
) -> None:
    """Emit auth/cache startup logs without exposing secrets."""
    if token_source is None:
        log.info(
            "No Hugging Face env token configured; relying on saved login or anonymous access."
        )
    else:
        log.info("Using Hugging Face env token from %s.", token_source)

    if hf_cache_dir is None:
        log.info("Using Hugging Face cache dir: library default")
    else:
        log.info("Using Hugging Face cache dir: %s", hf_cache_dir)


def build_gene_vocab(
    load_dataset_kwargs: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Tahoe-100M gene metadata and return ``(gene_ids, token_to_col_array)``.

    ``gene_ids``  — 1-D string array of Ensembl IDs, ordered by token_id.
    ``token_to_col_array`` — int64 array of length ``max_token_id + 1``.
        Maps ``token_id → column index`` for vectorised densification.
        Unmapped positions hold ``-1``.
    """
    load_dataset_kwargs = dict(load_dataset_kwargs or {})
    t0 = time.monotonic()
    log.info("Loading gene metadata from %s ...", DATASET_NAME)
    gene_meta = load_dataset(
        DATASET_NAME,
        name="gene_metadata",
        split="train",
        **load_dataset_kwargs,
    )
    log.info("Gene metadata load finished in %.1fs", time.monotonic() - t0)
    df = pd.DataFrame(gene_meta).dropna(subset=["token_id", "ensembl_id"])
    df = df.drop_duplicates(subset="ensembl_id", keep="first")
    df = df.sort_values("token_id").reset_index(drop=True)

    token_ids = df["token_id"].astype(int).values
    gene_ids = df["ensembl_id"].astype(str).values
    assert len(token_ids) > 0, "Gene vocabulary is empty after filtering"
    assert np.all(np.diff(token_ids) > 0), "token_ids are not strictly monotonic"

    # Build lookup array: token_id → column index.  -1 = unmapped.
    max_tok = int(token_ids.max())
    token_to_col = np.full(max_tok + 1, -1, dtype=np.int64)
    for col_idx, tok in enumerate(token_ids):
        token_to_col[tok] = col_idx

    n_genes = len(gene_ids)
    log.info("Gene vocab ready: %d genes, max token_id=%d", n_genes, max_tok)
    return gene_ids, token_to_col


# ---------------------------------------------------------------------------
# Ordered streaming helpers
# ---------------------------------------------------------------------------

def _densify_record(
    record: dict,
    n_genes: int,
    token_to_col: np.ndarray,
) -> tuple[PseudobulkGroupKey, np.ndarray] | None:
    """Parse one HF streaming record into (group_key, dense_expr).

    Returns ``None`` for records that should be skipped (missing keys).
    Raises on unexpected data (unknown token, non-finite expression).
    """
    group_key = _group_key_from_record(record)
    if group_key is None:
        return None

    genes = record["genes"]
    exprs = record["expressions"]

    # Sentinel: if the first expression is negative, drop the first pair.
    if exprs and exprs[0] < 0:
        genes = genes[1:]
        exprs = exprs[1:]

    genes_arr = np.asarray(genes, dtype=np.int64)
    exprs_arr = np.asarray(exprs, dtype=np.float32)

    assert len(genes_arr) == len(exprs_arr), (
        f"genes/expressions length mismatch: {len(genes_arr)} vs {len(exprs_arr)}"
    )

    # Validate token IDs are in vocab.
    if genes_arr.size > 0:
        if genes_arr.min() < 0:
            bad = genes_arr[genes_arr < 0]
            raise ValueError(
                f"Negative token IDs are invalid: {bad[:5].tolist()}"
            )
        max_tok_in_record = genes_arr.max()
        if max_tok_in_record >= len(token_to_col):
            bad = genes_arr[genes_arr >= len(token_to_col)]
            raise ValueError(
                f"Token IDs {bad[:5].tolist()} exceed vocab size ({len(token_to_col)})"
            )
        col_indices = token_to_col[genes_arr]
        unmapped = col_indices == -1
        if unmapped.any():
            bad = genes_arr[unmapped]
            raise ValueError(
                f"Unknown token IDs not in gene vocab: {bad[:5].tolist()}"
            )
        if len(col_indices) != len(np.unique(col_indices)):
            seen = set()
            dupes = []
            for g, c in zip(genes_arr, col_indices):
                if c in seen:
                    dupes.append(int(g))
                seen.add(c)
            raise ValueError(
                f"Duplicate gene tokens in record for "
                f"{group_key.cell_line_id}/{group_key.drug_key}: {dupes[:5]}"
            )
    else:
        col_indices = np.array([], dtype=np.int64)

    if not np.all(np.isfinite(exprs_arr)):
        raise ValueError(
            f"Non-finite expression values in record for "
            f"{group_key.cell_line_id}/{group_key.drug_key}"
        )

    dense = np.zeros(n_genes, dtype=np.float32)
    if col_indices.size > 0:
        dense[col_indices] = exprs_arr
    return group_key, dense


_WORKER_N_GENES: int | None = None
_WORKER_TOKEN_TO_COL: np.ndarray | None = None


def _materialize_record(record: dict) -> dict:
    """Copy a streaming row into a plain dict before multiprocessing dispatch."""
    return {key: value for key, value in record.items()}


def _stringify_record_value(record: dict, field_name: str) -> str:
    """Coerce metadata values to strings while treating missing values as empty."""
    value = record.get(field_name, "")
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return str(value)


def _group_key_from_record(record: dict) -> PseudobulkGroupKey | None:
    """Build the replicate-aware pseudobulk key for one Tahoe record."""
    cell_line_id = _stringify_record_value(record, "cell_line_id")
    drug_key = _stringify_record_value(record, "drugname_drugconc")
    if not cell_line_id or not drug_key:
        return None

    return PseudobulkGroupKey(
        cell_line_id=cell_line_id,
        drug_key=drug_key,
        sample=_stringify_record_value(record, "sample"),
        plate=_stringify_record_value(record, "plate"),
        barcode_sub_lib_id=_stringify_record_value(record, "BARCODE_SUB_LIB_ID"),
    )


def _init_densify_worker(n_genes: int, token_to_col: np.ndarray) -> None:
    """Store densification state once per worker process."""
    global _WORKER_N_GENES, _WORKER_TOKEN_TO_COL
    _WORKER_N_GENES = n_genes
    _WORKER_TOKEN_TO_COL = token_to_col


def _densify_record_worker(record: dict) -> tuple[PseudobulkGroupKey, np.ndarray] | None:
    """Worker entrypoint for order-preserving parallel densification."""
    assert _WORKER_N_GENES is not None
    assert _WORKER_TOKEN_TO_COL is not None
    return _densify_record(record, _WORKER_N_GENES, _WORKER_TOKEN_TO_COL)


def _get_mp_context():
    """Prefer ``fork`` when available so tests and CLI work on Unix."""
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def _iter_selected_records(
    densified_records,
    sample_size: int | None,
    cell_lines_whitelist: set[str] | None,
):
    """Apply whitelist and sample-size limits after densification."""
    if sample_size is not None and sample_size <= 0:
        return

    emitted = 0
    for result in densified_records:
        if result is None:
            continue

        group_key, _ = result
        if (
            cell_lines_whitelist is not None
            and group_key.cell_line_id not in cell_lines_whitelist
        ):
            continue

        yield result
        emitted += 1
        if sample_size is not None and emitted >= sample_size:
            break


def _iter_ordered_densified_records(
    records,
    n_genes: int,
    token_to_col: np.ndarray,
    sample_size: int | None = None,
    cell_lines_whitelist: set[str] | None = None,
    num_workers: int = 0,
):
    """Yield densified cells in source order regardless of worker count."""
    plain_records = (_materialize_record(record) for record in records)

    if num_workers == 0:
        densified_records = (
            _densify_record(record, n_genes, token_to_col)
            for record in plain_records
        )
        yield from _iter_selected_records(
            densified_records,
            sample_size=sample_size,
            cell_lines_whitelist=cell_lines_whitelist,
        )
        return

    ctx = _get_mp_context()
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_densify_worker,
        initargs=(n_genes, token_to_col),
    ) as pool:
        densified_records = pool.imap(
            _densify_record_worker,
            plain_records,
            chunksize=32,
        )
        yield from _iter_selected_records(
            densified_records,
            sample_size=sample_size,
            cell_lines_whitelist=cell_lines_whitelist,
        )


def _log_first_record_latency(records, started_at: float, stream_name: str):
    """Log when the first streamed record arrives to make slow startup visible."""
    first_record_seen = False
    for record in records:
        if not first_record_seen:
            log.info(
                "First record received from %s after %.1fs",
                stream_name,
                time.monotonic() - started_at,
            )
            first_record_seen = True
        yield record


# ---------------------------------------------------------------------------
# Pseudobulk pipeline
# ---------------------------------------------------------------------------

def run_pseudobulk(
    iterable,
    n_genes: int,
    block_size: int,
    progress_every: int = 25_000,
) -> dict[PseudobulkGroupKey, BlockAccumulator]:
    """Consume an iterable of (group_key, dense_expr) and accumulate.

    Works with both a raw iterable and an iterable of pre-batched tuples.
    """
    accumulators: dict[PseudobulkGroupKey, BlockAccumulator] = {}
    n_processed = 0
    t0 = time.monotonic()

    for item in iterable:
        # DataLoader batches come as lists of tuples; a raw iterable yields
        # single tuples.
        if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple)):
            batch = item
        else:
            batch = [item]

        for record in batch:
            group_key, dense = record
            if isinstance(dense, torch.Tensor):
                dense = dense.numpy()
            key = group_key
            acc = accumulators.get(key)
            if acc is None:
                acc = BlockAccumulator(n_genes=n_genes, block_size=block_size)
                accumulators[key] = acc
            acc.add_cell(dense)
            n_processed += 1

            if progress_every and n_processed % progress_every == 0:
                elapsed = time.monotonic() - t0
                rate = n_processed / elapsed if elapsed > 0 else 0
                log.info(
                    "Processed %d cells (%.0f cells/s), %d unique replicate-aware groups",
                    n_processed,
                    rate,
                    len(accumulators),
                )

    elapsed = time.monotonic() - t0
    log.info(
        "Streaming complete: %d cells in %.1fs (%.0f cells/s), %d groups",
        n_processed,
        elapsed,
        n_processed / elapsed if elapsed > 0 else 0,
        len(accumulators),
    )
    return accumulators


def write_outputs(
    accumulators: dict[PseudobulkGroupKey, BlockAccumulator],
    gene_ids: np.ndarray,
    output_dir: Path,
    run_meta: dict,
) -> list[Path]:
    """Write per-cell-line .h5ad files.  Returns list of paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_genes = len(gene_ids)

    # Group by cell line.
    by_cell_line: dict[str, list[tuple[PseudobulkGroupKey, BlockAccumulator]]] = {}
    n_excluded_total = 0
    for group_key, acc in accumulators.items():
        if acc.n_complete_blocks == 0:
            n_excluded_total += 1
            continue
        by_cell_line.setdefault(group_key.cell_line_id, []).append((group_key, acc))

    if not by_cell_line:
        raise RuntimeError(
            "No replicate-aware groups had enough cells for even one complete "
            f"block of size {run_meta.get('block_size', '?')}. "
            f"Total groups seen: {len(accumulators)}, all excluded."
        )

    if n_excluded_total > 0:
        # Report the most under-represented groups.
        shortfalls = sorted(
            (
                (
                    acc.n_cells_seen,
                    group_key.cell_line_id,
                    group_key.drug_key,
                    group_key.sample,
                    group_key.plate,
                    group_key.barcode_sub_lib_id,
                )
                for group_key, acc in accumulators.items()
                if acc.n_complete_blocks == 0
            ),
            reverse=True,
        )
        log.warning(
            "%d replicate-aware groups excluded (< %d cells). "
            "Top shortfalls: %s",
            n_excluded_total,
            run_meta.get("block_size", "?"),
            shortfalls[:10],
        )

    written: list[Path] = []
    for cell_line_id, pairs in sorted(by_cell_line.items()):
        pairs.sort(key=lambda t: t[0])
        group_keys = [group_key for group_key, _ in pairs]
        accs = [a for _, a in pairs]

        X = np.stack([a.finalize() for a in accs], axis=0)
        assert X.shape == (len(group_keys), n_genes), (
            f"Shape mismatch for {cell_line_id}: {X.shape} vs "
            f"({len(group_keys)}, {n_genes})"
        )
        assert np.all(np.isfinite(X)), f"Non-finite values in pseudobulk for {cell_line_id}"

        obs = pd.DataFrame(
            {
                "drugname_drugconc": [g.drug_key for g in group_keys],
                "sample": [g.sample for g in group_keys],
                "plate": [g.plate for g in group_keys],
                "BARCODE_SUB_LIB_ID": [g.barcode_sub_lib_id for g in group_keys],
                "replicate_id": [g.replicate_id for g in group_keys],
                "n_cells_total": [a.n_cells_seen for a in accs],
                "n_complete_blocks": [a.n_complete_blocks for a in accs],
                "n_cells_used": [a.n_cells_used for a in accs],
                "n_cells_dropped": [a.n_cells_dropped for a in accs],
            },
            index=pd.Index([g.obs_index for g in group_keys], name=OBS_INDEX_NAME),
        )
        var = pd.DataFrame(index=pd.Index(gene_ids, name="ensembl_id"))

        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.uns["cell_line_id"] = cell_line_id
        adata.uns["pseudobulk_groupby_columns"] = list(GROUPBY_OBS_COLUMNS)
        for k, v in run_meta.items():
            adata.uns[k] = v

        path = output_dir / f"{cell_line_id}.h5ad"
        adata.write_h5ad(path)
        log.info(
            "Wrote %s  — shape %s (%d replicate-aware groups × %d genes)",
            path,
            X.shape,
            X.shape[0],
            X.shape[1],
        )
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pseudobulk"),
        help="Directory for per-cell-line .h5ad files (default: data/pseudobulk)",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=200,
        help="Number of cells per pseudobulk block (default: 200)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Order-preserving densification workers (default: min(32, cpu_count))",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Max cells to stream (default: full dataset)",
    )
    p.add_argument(
        "--cell-lines",
        nargs="+",
        default=None,
        help="Optional whitelist of cell_line_id values",
    )
    p.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory (for example, a scratch disk)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=25_000,
        help="Log progress every N cells (default: 25000)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test: sample_size=2000, writes to a temp dir",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.smoke:
        args.sample_size = 2000
        args.output_dir = Path(tempfile.mkdtemp(prefix="pseudobulk_smoke_"))
        # Use fewer workers for a quick smoke test.
        args.num_workers = min(args.num_workers, 4)
        log.info("SMOKE TEST mode: sample_size=2000, output_dir=%s", args.output_dir)

    assert args.block_size > 0, f"--block-size must be > 0, got {args.block_size}"
    assert args.num_workers >= 0, f"--num-workers must be >= 0, got {args.num_workers}"
    assert args.sample_size is None or args.sample_size >= 0, (
        f"--sample-size must be >= 0, got {args.sample_size}"
    )

    hf_load_kwargs, token_source = _resolve_hf_load_kwargs(args.hf_cache_dir)
    _log_hf_startup_config(token_source, args.hf_cache_dir)

    # ---- Gene vocab ----
    gene_ids, token_to_col = build_gene_vocab(load_dataset_kwargs=hf_load_kwargs)
    n_genes = len(gene_ids)

    # ---- Build ordered stream ----
    cell_lines_whitelist = set(args.cell_lines) if args.cell_lines else None
    stream_open_t0 = time.monotonic()
    log.info("Opening streaming train split from %s ...", DATASET_NAME)
    raw_records = load_dataset(
        DATASET_NAME,
        streaming=True,
        split="train",
        **hf_load_kwargs,
    )
    log.info(
        "Streaming train split initialized in %.1fs; waiting for first record ...",
        time.monotonic() - stream_open_t0,
    )
    raw_records = _log_first_record_latency(
        raw_records,
        started_at=stream_open_t0,
        stream_name=f"{DATASET_NAME} train split",
    )
    ordered_stream = _iter_ordered_densified_records(
        raw_records,
        n_genes=n_genes,
        token_to_col=token_to_col,
        sample_size=args.sample_size,
        cell_lines_whitelist=cell_lines_whitelist,
        num_workers=args.num_workers,
    )

    # ---- Accumulate ----
    accumulators = run_pseudobulk(
        ordered_stream,
        n_genes=n_genes,
        block_size=args.block_size,
        progress_every=args.progress_every,
    )

    # ---- Write outputs ----
    run_meta = {
        "block_size": args.block_size,
        "source_dataset": DATASET_NAME,
        "sample_size": args.sample_size if args.sample_size is not None else "full",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    written = write_outputs(accumulators, gene_ids, args.output_dir, run_meta)

    # ---- Summary ----
    total_cells = sum(a.n_cells_seen for a in accumulators.values())
    total_used = sum(a.n_cells_used for a in accumulators.values())
    total_dropped = sum(a.n_cells_dropped for a in accumulators.values())
    pairs_with_blocks = sum(
        1 for a in accumulators.values() if a.n_complete_blocks > 0
    )
    pairs_without = len(accumulators) - pairs_with_blocks

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Total cells processed   : %d", total_cells)
    log.info("  Cells used in blocks    : %d", total_used)
    log.info("  Cells dropped (remainder): %d", total_dropped)
    log.info("  Unique replicate-aware groups: %d", len(accumulators))
    log.info("    with ≥1 block         : %d", pairs_with_blocks)
    log.info("    excluded (0 blocks)   : %d", pairs_without)
    log.info("  Cell lines written      : %d", len(written))
    log.info("  Output directory        : %s", args.output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
