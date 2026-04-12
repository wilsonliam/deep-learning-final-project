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
Accumulator memory scales with the number of replicate-aware groups that are
resident at once. This CLI now streams one cell line at a time so the working
set stays bounded by a single cell line's active groups rather than the full
dataset.

Dependencies: datasets, anndata, numpy, pandas, torch, scipy, tqdm
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

DATASET_NAME = "tahoebio/Tahoe-100M"
HF_TOKEN_ENV_VAR = "HF_TOKEN"
HF_TOKEN_ENV_VAR_LEGACY = "HUGGING_FACE_HUB_TOKEN"
HF_HUB_ENABLE_HF_TRANSFER_ENV_VAR = "HF_HUB_ENABLE_HF_TRANSFER"
HF_XET_HIGH_PERFORMANCE_ENV_VAR = "HF_XET_HIGH_PERFORMANCE"
HF_HUB_ETAG_TIMEOUT_ENV_VAR = "HF_HUB_ETAG_TIMEOUT"
HF_HUB_DOWNLOAD_TIMEOUT_ENV_VAR = "HF_HUB_DOWNLOAD_TIMEOUT"
HF_DEBUG_ENV_VAR = "HF_DEBUG"
GROUPBY_OBS_COLUMNS = (
    "drugname_drugconc",
    "sample",
    "plate",
    "BARCODE_SUB_LIB_ID",
)
OBS_INDEX_NAME = "pseudobulk_group"
REQUIRED_SAMPLE_METADATA_COLUMNS = (
    "sample",
    "drug",
    "plate",
    "drugname_drugconc",
)
REQUIRED_CELL_LINE_METADATA_COLUMNS = ("Cell_ID_Cellosaur",)
REQUIRED_EXPRESSION_STREAM_COLUMNS = (
    "cell_line_id",
    "drug",
    "sample",
    "plate",
    "BARCODE_SUB_LIB_ID",
    "genes",
    "expressions",
)
PROGRESS_BAR_REFRESH_SECONDS = 1.0
PROGRESS_PHASE_LABEL = "Phase 4/4: streaming accumulation"


@dataclass(frozen=True, order=True)
class PseudobulkGroupKey:
    """Replicate-aware aggregation key for one pseudobulk row."""

    cell_line_id: str
    drug_key: str
    drug: str = ""
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


@dataclass(frozen=True)
class SampleMetadataEntry:
    """Canonical sample-level metadata used to enrich streamed expression rows."""

    sample: str
    drug: str
    plate: str
    drugname_drugconc: str


@dataclass(frozen=True)
class AccumulatorSummary:
    """Aggregate counts for one set of replicate-aware accumulators."""

    total_cells_processed: int
    total_cells_used: int
    total_cells_dropped: int
    total_groups: int
    groups_with_blocks: int
    groups_without_blocks: int
    shortfalls: tuple[tuple[int, str, str, str, str, str], ...] = ()


_TQDM_FACTORY = _tqdm


def _format_duration(seconds: float | None) -> str:
    """Render a compact duration string for logs and ETAs."""
    if seconds is None or not math.isfinite(seconds):
        return "unknown"

    whole_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_progress_log_message(
    processed_cells: int,
    total_cells: int | None,
    total_is_estimated: bool,
    n_groups: int,
    elapsed_seconds: float,
) -> str:
    """Build a human-readable progress log line."""
    rate = processed_cells / elapsed_seconds if elapsed_seconds > 0 else 0.0
    elapsed_text = _format_duration(elapsed_seconds)
    groups_text = f"{n_groups:,} groups"

    if total_cells is None:
        return (
            f"Processed {processed_cells:,} cells "
            f"(elapsed {elapsed_text}, {rate:.0f} cells/s, {groups_text}, total unknown)"
        )

    total_text = f"{total_cells:,}"
    if total_is_estimated:
        total_text = f"~{total_text}"

    progress_percent = (processed_cells / total_cells * 100.0) if total_cells > 0 else 100.0
    remaining_cells = max(total_cells - processed_cells, 0)
    eta_seconds = (
        remaining_cells / rate
        if rate > 0 and processed_cells < total_cells
        else 0.0 if processed_cells >= total_cells else None
    )
    parts = [
        f"Processed {processed_cells:,}/{total_text} cells",
        f"{progress_percent:.2f}%",
        f"elapsed {elapsed_text}",
        f"{rate:.0f} cells/s",
        f"ETA {_format_duration(eta_seconds)}",
        groups_text,
    ]
    if total_is_estimated:
        parts.append("estimated total")
    return f"{parts[0]} ({', '.join(parts[1:])})"


def _describe_progress_total(
    total_cells: int | None,
    total_is_estimated: bool,
) -> str:
    """Describe the total used for progress accounting."""
    if total_cells is None:
        return "open-ended total"
    if total_is_estimated:
        return f"estimated total {total_cells:,} cells"
    return f"total {total_cells:,} cells"


def _describe_progress_mode(progress_reporter: ProgressReporter) -> str:
    """Describe the active progress reporter for startup logs."""
    if isinstance(progress_reporter, TqdmProgressReporter):
        return "live bar"
    if isinstance(progress_reporter, LogProgressReporter):
        return "log progress"
    return "progress off"


def _stderr_supports_live_progress(stream=None) -> bool:
    """Return whether stderr supports a live single-line progress display."""
    stream = sys.stderr if stream is None else stream
    return bool(stream is not None and hasattr(stream, "isatty") and stream.isatty())


def _resolve_progress_total_cells(
    *,
    remaining_sample_size: int | None,
    progress_total_cells: int | None,
    target_cell_line_count: int,
) -> tuple[int | None, bool]:
    """Choose the per-cell-line total used for percent/ETA math."""
    if remaining_sample_size is not None:
        return remaining_sample_size, False
    if target_cell_line_count == 1 and progress_total_cells is not None:
        return progress_total_cells, False
    return None, False


class ProgressReporter:
    """Minimal interface for streaming progress displays."""

    def update(self, processed_cells: int, n_groups: int) -> None:
        return None

    def close(self, processed_cells: int, n_groups: int) -> None:
        return None


class NullProgressReporter(ProgressReporter):
    """Reporter that intentionally does nothing."""


class LogProgressReporter(ProgressReporter):
    """Periodic progress logger for non-TTY runs."""

    def __init__(
        self,
        *,
        total_cells: int | None,
        total_is_estimated: bool,
        progress_every: int,
    ) -> None:
        self.total_cells = total_cells
        self.total_is_estimated = total_is_estimated
        self.progress_every = progress_every
        self._started_at = time.monotonic()
        self._last_logged_processed = 0

    def _emit(self, processed_cells: int, n_groups: int) -> None:
        elapsed_seconds = time.monotonic() - self._started_at
        log.info(
            "%s",
            _format_progress_log_message(
                processed_cells=processed_cells,
                total_cells=self.total_cells,
                total_is_estimated=self.total_is_estimated,
                n_groups=n_groups,
                elapsed_seconds=elapsed_seconds,
            ),
        )

    def update(self, processed_cells: int, n_groups: int) -> None:
        if self.progress_every <= 0:
            return
        if processed_cells - self._last_logged_processed < self.progress_every:
            return
        self._last_logged_processed = processed_cells
        self._emit(processed_cells, n_groups)

    def close(self, processed_cells: int, n_groups: int) -> None:
        if self.progress_every <= 0:
            return
        if processed_cells == self._last_logged_processed:
            return
        self._last_logged_processed = processed_cells
        self._emit(processed_cells, n_groups)


class TqdmProgressReporter(ProgressReporter):
    """Single-line live tqdm bar for interactive runs."""

    def __init__(
        self,
        *,
        total_cells: int | None,
        total_is_estimated: bool,
        progress_every: int,
        stream=None,
        tqdm_factory=None,
    ) -> None:
        self.total_cells = total_cells
        self.total_is_estimated = total_is_estimated
        self.progress_every = progress_every
        self._stream = sys.stderr if stream is None else stream
        self._started_at = time.monotonic()
        self._last_seen_processed = 0
        self._pending_delta = 0
        self._last_rendered_at = self._started_at
        self._last_groups = 0
        desc = PROGRESS_PHASE_LABEL
        if self.total_is_estimated:
            desc += " (estimated total)"
        factory = _TQDM_FACTORY if tqdm_factory is None else tqdm_factory
        assert factory is not None, "tqdm factory is required for TqdmProgressReporter"
        self._bar = factory(
            total=self.total_cells,
            desc=desc,
            unit="cells",
            dynamic_ncols=True,
            leave=False,
            file=self._stream,
        )

    def _flush(self, n_groups: int, now: float) -> None:
        if self._pending_delta > 0:
            self._bar.update(self._pending_delta)
            self._pending_delta = 0
        postfix_parts = [f"groups={n_groups:,}"]
        if self.total_is_estimated:
            postfix_parts.append("total=est.")
        self._bar.set_postfix_str(", ".join(postfix_parts), refresh=False)
        self._bar.refresh()
        self._last_groups = n_groups
        self._last_rendered_at = now

    def update(self, processed_cells: int, n_groups: int) -> None:
        delta = processed_cells - self._last_seen_processed
        if delta < 0:
            return
        self._last_seen_processed = processed_cells
        self._pending_delta += delta
        now = time.monotonic()
        refresh_due = (
            self._pending_delta >= self.progress_every
            if self.progress_every > 0
            else self._pending_delta > 0
        )
        time_due = (now - self._last_rendered_at) >= PROGRESS_BAR_REFRESH_SECONDS
        if refresh_due or time_due:
            self._flush(n_groups, now)

    def close(self, processed_cells: int, n_groups: int) -> None:
        delta = processed_cells - self._last_seen_processed
        if delta > 0:
            self._pending_delta += delta
            self._last_seen_processed = processed_cells
        if self._pending_delta > 0 or n_groups != self._last_groups:
            self._flush(n_groups, time.monotonic())
        self._bar.close()


def _resolve_progress_mode(progress: str, *, stream=None) -> str:
    """Choose the concrete progress implementation for this run."""
    if progress == "off":
        return "off"
    if progress == "log":
        return "log"
    if progress == "bar":
        if _TQDM_FACTORY is None:
            log.warning("tqdm is unavailable; falling back to log progress.")
            return "log"
        return "bar"

    if _stderr_supports_live_progress(stream=stream) and _TQDM_FACTORY is not None:
        return "bar"
    return "log"


def _build_progress_reporter(
    *,
    progress: str,
    progress_every: int,
    total_cells: int | None,
    total_is_estimated: bool,
    stream=None,
) -> ProgressReporter:
    """Construct the chosen progress reporter."""
    resolved_mode = _resolve_progress_mode(progress, stream=stream)
    if resolved_mode == "off":
        return NullProgressReporter()
    if resolved_mode == "bar":
        return TqdmProgressReporter(
            total_cells=total_cells,
            total_is_estimated=total_is_estimated,
            progress_every=progress_every,
            stream=stream,
        )
    return LogProgressReporter(
        total_cells=total_cells,
        total_is_estimated=total_is_estimated,
        progress_every=progress_every,
    )

# ---------------------------------------------------------------------------
# CellLineMatrixAccumulator
# ---------------------------------------------------------------------------

class CellLineMatrixAccumulator:
    """Memory-efficient accumulator for all pseudobulk groups of one cell line.

    Uses a single disk-backed ``numpy.memmap`` array (rows = treatment
    conditions, columns = genes) to store running sums, and a compact 1-D
    int32 array in RAM to track per-condition cell counts.  The on-disk file
    grows dynamically as new conditions are discovered.

    Math at :meth:`finalize` time:
        pseudobulk = total_sum / (total_cells / block_size)

    This is equivalent to averaging ``block_size``-cell blocks, but uses all
    cells (including any remainder) and avoids the per-block bookkeeping of the
    old ``BlockAccumulator``.  Only conditions that accumulated at least
    ``block_size`` cells are retained.
    """

    def __init__(self, n_genes: int, block_size: int, initial_capacity: int = 1000):
        assert n_genes > 0, f"n_genes must be > 0, got {n_genes}"
        assert block_size > 0, f"block_size must be > 0, got {block_size}"
        self.n_genes = n_genes
        self.block_size = block_size
        self.capacity = initial_capacity
        self.n_active = 0

        self.key_to_idx: dict[PseudobulkGroupKey, int] = {}
        self.keys: list[PseudobulkGroupKey] = []

        self.counts = np.zeros(self.capacity, dtype=np.int32)

        fd, self.filename = tempfile.mkstemp(
            prefix="pseudobulk_matrix_", suffix=".dat"
        )
        os.close(fd)
        self.sums = np.memmap(
            self.filename,
            dtype=np.float32,
            mode="w+",
            shape=(self.capacity, self.n_genes),
        )

    def add_cell(self, key: PseudobulkGroupKey, expr: np.ndarray) -> None:
        """Accumulate one cell's dense expression vector into its condition row."""
        idx = self.key_to_idx.get(key)
        if idx is None:
            idx = self.n_active
            if idx >= self.capacity:
                self._expand()
            self.key_to_idx[key] = idx
            self.keys.append(key)
            self.n_active += 1

        self.sums[idx] += expr
        self.counts[idx] += 1

    def _expand(self) -> None:
        """Double the capacity of the memory-mapped sums array on disk."""
        new_capacity = self.capacity * 2
        self.sums.flush()
        del self.sums

        with open(self.filename, "r+b") as f:
            f.truncate(new_capacity * self.n_genes * np.dtype(np.float32).itemsize)

        self.sums = np.memmap(
            self.filename,
            dtype=np.float32,
            mode="r+",
            shape=(new_capacity, self.n_genes),
        )

        new_counts = np.zeros(new_capacity, dtype=np.int32)
        new_counts[: self.capacity] = self.counts
        self.counts = new_counts
        self.capacity = new_capacity

    def finalize(
        self,
    ) -> tuple[np.ndarray | None, list[PseudobulkGroupKey] | None, np.ndarray | None]:
        """Return ``(X, keys, counts)`` for conditions that reached block_size.

        Returns ``(None, None, None)`` when no conditions qualify.

        ``X`` has shape ``(n_valid, n_genes)`` with dtype ``float32``.
        Math applied: ``total_sum / (total_cells / block_size)``.
        """
        valid_indices = [
            i for i in range(self.n_active) if self.counts[i] >= self.block_size
        ]
        if not valid_indices:
            return None, None, None

        valid_counts = self.counts[valid_indices]
        scaling_factors = self.block_size / valid_counts
        final_X = np.array(self.sums[valid_indices]) * scaling_factors[:, np.newaxis]
        final_keys = [self.keys[i] for i in valid_indices]

        return final_X.astype(np.float32), final_keys, valid_counts

    def cleanup(self) -> None:
        """Flush and delete the temporary backing file from disk."""
        if hasattr(self, "sums"):
            del self.sums
        if os.path.exists(self.filename):
            os.remove(self.filename)


# ---------------------------------------------------------------------------
# Gene vocabulary
# ---------------------------------------------------------------------------

def _resolve_hf_load_kwargs(
    hf_cache_dir: Path | None = None,
) -> tuple[dict[str, object], str | None]:
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

    load_kwargs: dict[str, object] = {}
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

    for env_name in (
        HF_XET_HIGH_PERFORMANCE_ENV_VAR,
        HF_HUB_ETAG_TIMEOUT_ENV_VAR,
        HF_HUB_DOWNLOAD_TIMEOUT_ENV_VAR,
        HF_DEBUG_ENV_VAR,
    ):
        log.info("%s=%s", env_name, os.getenv(env_name, "<unset>"))

    if os.getenv(HF_HUB_ENABLE_HF_TRANSFER_ENV_VAR):
        log.warning(
            "%s is deprecated; current Hugging Face Hub clients use Xet instead "
            "of hf_transfer. Prefer %s=1 for high-performance transfers.",
            HF_HUB_ENABLE_HF_TRANSFER_ENV_VAR,
            HF_XET_HIGH_PERFORMANCE_ENV_VAR,
        )


def _stringify_scalar_value(value) -> str:
    """Coerce metadata scalars to strings while treating missing values as empty."""
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return str(value).strip()


def _validate_expression_stream_columns(
    available_columns: list[str],
    stream_name: str,
) -> None:
    """Fail fast when the live Tahoe stream no longer matches the expected schema."""
    missing = sorted(set(REQUIRED_EXPRESSION_STREAM_COLUMNS) - set(available_columns))
    if not missing:
        return

    raise RuntimeError(
        f"{stream_name} schema is incompatible: missing required columns "
        f"{missing}. Available columns: {sorted(available_columns)}"
    )


def _preflight_expression_stream_schema(records, stream_name: str) -> None:
    """Validate stream columns before the main pseudobulk pass begins."""
    t0 = time.monotonic()
    features = getattr(records, "features", None)
    if features is not None:
        available_columns = list(features.keys())
        _validate_expression_stream_columns(available_columns, stream_name)
        log.info(
            "Streaming schema preflight for %s succeeded from dataset features in %.1fs",
            stream_name,
            time.monotonic() - t0,
        )
        return

    log.warning(
        "Streaming schema unavailable via dataset features for %s; peeking the first record.",
        stream_name,
    )
    iterator = iter(records)
    try:
        first_record = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(
            f"{stream_name} produced no rows during schema preflight."
        ) from exc

    if not hasattr(first_record, "keys"):
        raise RuntimeError(
            f"{stream_name} yielded {type(first_record).__name__} during schema "
            "preflight; expected a mapping-like row."
        )

    _validate_expression_stream_columns(list(first_record.keys()), stream_name)
    log.info(
        "Streaming schema preflight for %s succeeded from the first record in %.1fs",
        stream_name,
        time.monotonic() - t0,
    )


def _build_expression_stream_load_kwargs(
    base_load_kwargs: dict[str, object],
    cell_lines_whitelist: set[str] | None,
) -> dict[str, object]:
    """Build the load kwargs for the main expression stream."""
    load_kwargs = dict(base_load_kwargs)
    load_kwargs["columns"] = list(REQUIRED_EXPRESSION_STREAM_COLUMNS)
    if cell_lines_whitelist:
        load_kwargs["filters"] = [
            ("cell_line_id", "in", sorted(cell_lines_whitelist)),
        ]
    return load_kwargs


def _log_expression_stream_load_plan(
    expression_load_kwargs: dict[str, object],
    cell_lines_whitelist: set[str] | None,
) -> None:
    """Summarize the selected stream shape before records start arriving."""
    log.info(
        "Streaming expression columns: %s",
        ", ".join(expression_load_kwargs["columns"]),
    )
    if cell_lines_whitelist:
        log.info(
            "Streaming expression filter: cell_line_id in %d value(s)",
            len(cell_lines_whitelist),
        )
    else:
        log.info("Streaming expression filter: none")


def build_sample_metadata_lookup(
    sample_metadata_rows,
) -> dict[str, SampleMetadataEntry]:
    """Validate sample metadata rows and build a lookup keyed by sample."""
    df = pd.DataFrame(sample_metadata_rows)
    missing_columns = sorted(
        set(REQUIRED_SAMPLE_METADATA_COLUMNS) - set(df.columns)
    )
    if missing_columns:
        raise RuntimeError(
            "sample_metadata schema is incompatible: missing required columns "
            f"{missing_columns}. Available columns: {sorted(df.columns.tolist())}"
        )

    df = df.loc[:, list(REQUIRED_SAMPLE_METADATA_COLUMNS)].copy()
    for column_name in REQUIRED_SAMPLE_METADATA_COLUMNS:
        df[column_name] = df[column_name].map(_stringify_scalar_value)

    empty_samples = df["sample"] == ""
    if empty_samples.any():
        raise RuntimeError(
            "sample_metadata contains empty sample keys; sample must be a unique "
            "non-empty join key."
        )

    duplicate_samples = sorted(
        df.loc[df["sample"].duplicated(keep=False), "sample"].unique().tolist()
    )
    if duplicate_samples:
        raise RuntimeError(
            "sample_metadata contains duplicate sample keys: "
            f"{duplicate_samples[:10]}"
        )

    for column_name in ("drug", "plate", "drugname_drugconc"):
        empty_rows = df[column_name] == ""
        if empty_rows.any():
            samples = sorted(df.loc[empty_rows, "sample"].tolist())
            raise RuntimeError(
                f"sample_metadata contains empty {column_name} values for samples "
                f"{samples[:10]}"
            )

    return {
        row["sample"]: SampleMetadataEntry(
            sample=row["sample"],
            drug=row["drug"],
            plate=row["plate"],
            drugname_drugconc=row["drugname_drugconc"],
        )
        for row in df.to_dict(orient="records")
    }


def load_sample_metadata_lookup(
    load_dataset_kwargs: dict[str, object] | None = None,
) -> dict[str, SampleMetadataEntry]:
    """Load Tahoe sample metadata and build a validated lookup by sample."""
    load_dataset_kwargs = dict(load_dataset_kwargs or {})
    t0 = time.monotonic()
    log.info("Loading sample metadata from %s ...", DATASET_NAME)
    sample_metadata = load_dataset(
        DATASET_NAME,
        name="sample_metadata",
        split="train",
        **load_dataset_kwargs,
    )
    lookup = build_sample_metadata_lookup(sample_metadata)
    log.info(
        "Sample metadata load finished in %.1fs (%d samples)",
        time.monotonic() - t0,
        len(lookup),
    )
    return lookup


def _normalize_cell_line_ids(values) -> list[str]:
    """Normalize, validate, de-duplicate, and sort cell line identifiers."""
    normalized = [_stringify_scalar_value(value) for value in values]
    if any(value == "" for value in normalized):
        raise RuntimeError("cell_line_id values must be non-empty after normalization.")
    unique_ids = sorted(set(normalized))
    if not unique_ids:
        raise RuntimeError("No usable cell_line_id values were found.")
    return unique_ids


def build_target_cell_line_ids(cell_line_metadata_rows) -> list[str]:
    """Validate Tahoe cell-line metadata rows and return sorted target ids."""
    df = pd.DataFrame(cell_line_metadata_rows)
    missing_columns = sorted(
        set(REQUIRED_CELL_LINE_METADATA_COLUMNS) - set(df.columns)
    )
    if missing_columns:
        raise RuntimeError(
            "cell_line_metadata schema is incompatible: missing required columns "
            f"{missing_columns}. Available columns: {sorted(df.columns.tolist())}"
        )

    return _normalize_cell_line_ids(df["Cell_ID_Cellosaur"].tolist())


def load_target_cell_line_ids(
    load_dataset_kwargs: dict[str, object] | None = None,
) -> list[str]:
    """Load Tahoe cell_line_metadata and return sorted unique target ids."""
    load_dataset_kwargs = dict(load_dataset_kwargs or {})
    t0 = time.monotonic()
    log.info("Loading cell line metadata from %s ...", DATASET_NAME)
    cell_line_metadata = load_dataset(
        DATASET_NAME,
        name="cell_line_metadata",
        split="train",
        **load_dataset_kwargs,
    )
    cell_line_ids = build_target_cell_line_ids(cell_line_metadata)
    log.info(
        "Cell line metadata load finished in %.1fs (%d cell lines)",
        time.monotonic() - t0,
        len(cell_line_ids),
    )
    return cell_line_ids


def build_gene_vocab(
    load_dataset_kwargs: dict[str, object] | None = None,
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
    return _stringify_scalar_value(record.get(field_name, ""))


def _normalize_record_with_sample_metadata(
    record: dict,
    sample_metadata_by_sample: dict[str, SampleMetadataEntry],
) -> dict:
    """Enrich a streamed expression row with concentration-aware sample metadata."""
    sample = _stringify_record_value(record, "sample")
    if not sample:
        raise RuntimeError("Expression stream row is missing the required sample value.")

    sample_meta = sample_metadata_by_sample.get(sample)
    if sample_meta is None:
        raise RuntimeError(
            f"Expression stream row references sample {sample!r}, but no matching "
            "sample_metadata row was found."
        )

    stream_drug = _stringify_record_value(record, "drug")
    if not stream_drug:
        raise RuntimeError(
            f"Expression stream row for sample {sample!r} is missing the required drug value."
        )
    if stream_drug != sample_meta.drug:
        raise RuntimeError(
            f"Expression stream drug {stream_drug!r} disagrees with sample_metadata "
            f"drug {sample_meta.drug!r} for sample {sample!r}."
        )

    stream_plate = _stringify_record_value(record, "plate")
    if not stream_plate:
        raise RuntimeError(
            f"Expression stream row for sample {sample!r} is missing the required plate value."
        )
    if stream_plate != sample_meta.plate:
        raise RuntimeError(
            f"Expression stream plate {stream_plate!r} disagrees with sample_metadata "
            f"plate {sample_meta.plate!r} for sample {sample!r}."
        )

    record["sample"] = sample_meta.sample
    record["drug"] = sample_meta.drug
    record["plate"] = sample_meta.plate
    record["drugname_drugconc"] = sample_meta.drugname_drugconc
    return record


def _group_key_from_record(record: dict) -> PseudobulkGroupKey | None:
    """Build the replicate-aware pseudobulk key for one Tahoe record."""
    cell_line_id = _stringify_record_value(record, "cell_line_id")
    drug = _stringify_record_value(record, "drug")
    drug_key = _stringify_record_value(record, "drugname_drugconc")
    if not cell_line_id or not drug or not drug_key:
        return None

    return PseudobulkGroupKey(
        cell_line_id=cell_line_id,
        drug_key=drug_key,
        drug=drug,
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
    sample_metadata_by_sample: dict[str, SampleMetadataEntry],
    sample_size: int | None = None,
    cell_lines_whitelist: set[str] | None = None,
    num_workers: int = 0,
):
    """Yield densified cells in source order regardless of worker count."""
    plain_records = (
        _normalize_record_with_sample_metadata(
            _materialize_record(record),
            sample_metadata_by_sample,
        )
        for record in records
    )

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
    progress_reporter: ProgressReporter | None = None,
) -> CellLineMatrixAccumulator:
    """Consume an iterable of (group_key, dense_expr) and accumulate.

    Returns a single :class:`CellLineMatrixAccumulator` backed by a
    temporary memory-mapped file.  The caller is responsible for calling
    :meth:`CellLineMatrixAccumulator.cleanup` once the data has been saved.

    Works with both a raw iterable and an iterable of pre-batched tuples.
    """
    matrix_acc = CellLineMatrixAccumulator(n_genes=n_genes, block_size=block_size)
    n_processed = 0
    t0 = time.monotonic()
    progress_reporter = progress_reporter or NullProgressReporter()

    try:
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
                matrix_acc.add_cell(group_key, dense)
                n_processed += 1
                progress_reporter.update(
                    processed_cells=n_processed,
                    n_groups=matrix_acc.n_active,
                )
    finally:
        progress_reporter.close(
            processed_cells=n_processed,
            n_groups=matrix_acc.n_active,
        )

    elapsed = time.monotonic() - t0
    log.info(
        "Streaming complete: %d cells in %.1fs (%.0f cells/s), %d groups",
        n_processed,
        elapsed,
        n_processed / elapsed if elapsed > 0 else 0,
        matrix_acc.n_active,
    )
    return matrix_acc


def _summarize_accumulators(
    matrix_acc: CellLineMatrixAccumulator,
) -> AccumulatorSummary:
    """Compute aggregate counts and exclusion details from a CellLineMatrixAccumulator."""
    counts = matrix_acc.counts[: matrix_acc.n_active]
    block_size = matrix_acc.block_size

    valid_mask = counts >= block_size
    invalid_mask = ~valid_mask

    total_cells_processed = int(counts.sum())
    total_cells_used = int(counts[valid_mask].sum())
    total_cells_dropped = int(counts[invalid_mask].sum())
    groups_with_blocks = int(valid_mask.sum())

    shortfalls = tuple(
        sorted(
            (
                (
                    int(counts[i]),
                    matrix_acc.keys[i].cell_line_id,
                    matrix_acc.keys[i].drug_key,
                    matrix_acc.keys[i].sample,
                    matrix_acc.keys[i].plate,
                    matrix_acc.keys[i].barcode_sub_lib_id,
                )
                for i in range(matrix_acc.n_active)
                if counts[i] < block_size
            ),
            reverse=True,
        )
    )
    return AccumulatorSummary(
        total_cells_processed=total_cells_processed,
        total_cells_used=total_cells_used,
        total_cells_dropped=total_cells_dropped,
        total_groups=matrix_acc.n_active,
        groups_with_blocks=groups_with_blocks,
        groups_without_blocks=matrix_acc.n_active - groups_with_blocks,
        shortfalls=shortfalls,
    )


def _cell_line_output_path(output_dir: Path, cell_line_id: str) -> Path:
    """Return the final output path for one cell line."""
    return output_dir / f"{cell_line_id}.h5ad"


def write_cell_line_output(
    cell_line_id: str,
    matrix_acc: CellLineMatrixAccumulator,
    gene_ids: np.ndarray,
    output_dir: Path,
    run_meta: dict,
) -> Path | None:
    """Write a single cell line atomically, or return ``None`` when empty/sparse."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_genes = len(gene_ids)
    summary = _summarize_accumulators(matrix_acc)

    if summary.groups_with_blocks == 0:
        if summary.groups_without_blocks > 0:
            log.warning(
                "Cell line %s skipped: %d replicate-aware groups excluded (< %d cells). "
                "Top shortfalls: %s",
                cell_line_id,
                summary.groups_without_blocks,
                run_meta.get("block_size", "?"),
                summary.shortfalls[:10],
            )
        else:
            log.warning(
                "Cell line %s skipped: no expression rows produced any replicate-aware groups.",
                cell_line_id,
            )
        return None

    if summary.groups_without_blocks > 0:
        log.warning(
            "Cell line %s excluded %d replicate-aware groups (< %d cells). "
            "Top shortfalls: %s",
            cell_line_id,
            summary.groups_without_blocks,
            run_meta.get("block_size", "?"),
            summary.shortfalls[:10],
        )

    X, group_keys, valid_counts = matrix_acc.finalize()
    if X is None or group_keys is None:
        raise RuntimeError(
            f"Expected replicate-aware groups for cell line {cell_line_id}, but none were retained."
        )

    # Sort rows by group key for deterministic output ordering.
    sorted_order = sorted(range(len(group_keys)), key=lambda i: group_keys[i])
    group_keys = [group_keys[i] for i in sorted_order]
    valid_counts = valid_counts[sorted_order]
    X = X[sorted_order]

    assert X.shape == (len(group_keys), n_genes), (
        f"Shape mismatch for {cell_line_id}: {X.shape} vs "
        f"({len(group_keys)}, {n_genes})"
    )
    assert np.all(np.isfinite(X)), f"Non-finite values in pseudobulk for {cell_line_id}"

    block_size = matrix_acc.block_size
    obs = pd.DataFrame(
        {
            "drug": [group_key.drug for group_key in group_keys],
            "drugname_drugconc": [group_key.drug_key for group_key in group_keys],
            "sample": [group_key.sample for group_key in group_keys],
            "plate": [group_key.plate for group_key in group_keys],
            "BARCODE_SUB_LIB_ID": [group_key.barcode_sub_lib_id for group_key in group_keys],
            "replicate_id": [group_key.replicate_id for group_key in group_keys],
            "n_cells_total": valid_counts.tolist(),
            # n_complete_blocks is informational; the actual math uses all cells.
            "n_complete_blocks": (valid_counts // block_size).tolist(),
            # All cells are used in the new accumulation math (no remainder dropped).
            "n_cells_used": valid_counts.tolist(),
            "n_cells_dropped": [0] * len(group_keys),
        },
        index=pd.Index([group_key.obs_index for group_key in group_keys], name=OBS_INDEX_NAME),
    )
    var = pd.DataFrame(index=pd.Index(gene_ids, name="ensembl_id"))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["cell_line_id"] = cell_line_id
    adata.uns["pseudobulk_groupby_columns"] = list(GROUPBY_OBS_COLUMNS)
    for key, value in run_meta.items():
        adata.uns[key] = value

    final_path = _cell_line_output_path(output_dir, cell_line_id)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{cell_line_id}.",
        suffix=".h5ad",
        dir=output_dir,
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        adata.write_h5ad(temp_path)
        os.replace(temp_path, final_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    log.info(
        "Wrote %s  — shape %s (%d replicate-aware groups × %d genes)",
        final_path,
        X.shape,
        X.shape[0],
        X.shape[1],
    )
    return final_path


def write_outputs(
    matrix_acc: CellLineMatrixAccumulator,
    cell_line_id: str,
    gene_ids: np.ndarray,
    output_dir: Path,
    run_meta: dict,
) -> list[Path]:
    """Write a per-cell-line .h5ad file.  Returns list of paths written."""
    if matrix_acc.n_active == 0:
        raise RuntimeError(
            "No replicate-aware groups had enough cells for even one complete "
            f"block of size {run_meta.get('block_size', '?')}. "
            f"Total groups seen: {matrix_acc.n_active}, all excluded."
        )

    written: list[Path] = []
    path = write_cell_line_output(
        cell_line_id,
        matrix_acc,
        gene_ids,
        output_dir,
        run_meta,
    )
    if path is not None:
        written.append(path)

    if not written:
        raise RuntimeError(
            "No replicate-aware groups had enough cells for even one complete "
            f"block of size {run_meta.get('block_size', '?')}. "
            f"Total groups seen: {matrix_acc.n_active}, all excluded."
        )

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
        "--progress",
        choices=("auto", "bar", "log", "off"),
        default="auto",
        help="Progress display mode: auto, bar, log, or off (default: auto)",
    )
    p.add_argument(
        "--progress-total-cells",
        type=int,
        default=None,
        help="Optional override for progress percent/ETA total cell count",
    )
    p.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute cell-line outputs even when the final .h5ad already exists",
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
    assert args.progress_total_cells is None or args.progress_total_cells > 0, (
        f"--progress-total-cells must be > 0, got {args.progress_total_cells}"
    )

    hf_load_kwargs, token_source = _resolve_hf_load_kwargs(args.hf_cache_dir)
    _log_hf_startup_config(token_source, args.hf_cache_dir)

    # ---- Gene vocab ----
    log.info("Phase 1/4: loading gene metadata")
    gene_ids, token_to_col = build_gene_vocab(load_dataset_kwargs=hf_load_kwargs)
    n_genes = len(gene_ids)

    # ---- Sample metadata ----
    log.info("Phase 2/4: loading sample metadata")
    sample_metadata_by_sample = load_sample_metadata_lookup(
        load_dataset_kwargs=hf_load_kwargs,
    )
    log.info(
        "Resolving concentration-aware labels via sample_metadata for %d samples.",
        len(sample_metadata_by_sample),
    )

    if args.cell_lines:
        target_cell_line_ids = _normalize_cell_line_ids(args.cell_lines)
        log.info(
            "Using %d requested target cell lines from --cell-lines.",
            len(target_cell_line_ids),
        )
    else:
        target_cell_line_ids = load_target_cell_line_ids(
            load_dataset_kwargs=hf_load_kwargs,
        )
    log.info("Resolved %d target cell lines.", len(target_cell_line_ids))
    if args.progress_total_cells is not None and len(target_cell_line_ids) != 1:
        log.warning(
            "--progress-total-cells is ignored for multi-cell-line runs; "
            "inner progress is reported per active cell line."
        )

    # ---- Preflight and build ordered stream ----
    preflight_t0 = time.monotonic()
    stream_name = f"{DATASET_NAME} train split"
    log.info("Phase 3/4: schema preflight")
    log.info("Opening streaming train split from %s for schema preflight ...", DATASET_NAME)
    raw_schema_records = load_dataset(
        DATASET_NAME,
        streaming=True,
        split="train",
        **hf_load_kwargs,
    )
    _preflight_expression_stream_schema(raw_schema_records, stream_name=stream_name)
    log.info(
        "Streaming schema preflight finished in %.1fs",
        time.monotonic() - preflight_t0,
    )

    run_meta = {
        "block_size": args.block_size,
        "source_dataset": DATASET_NAME,
        "sample_size": args.sample_size if args.sample_size is not None else "full",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    total_target_cell_lines = len(target_cell_line_ids)
    total_cells_processed = 0
    total_cells_used = 0
    total_cells_dropped = 0
    total_groups_seen = 0
    total_groups_with_blocks = 0
    total_groups_without_blocks = 0
    written: list[Path] = []
    skipped_existing = 0
    skipped_sparse_or_empty = 0
    stopped_early_due_to_sample_size = False
    remaining_sample_size = args.sample_size

    log.info("Phase 4/4: streaming accumulation")
    for index, cell_line_id in enumerate(target_cell_line_ids, start=1):
        if remaining_sample_size is not None and remaining_sample_size <= 0:
            stopped_early_due_to_sample_size = True
            log.info(
                "Stopping before cell line %d/%d (%s): global --sample-size budget exhausted.",
                index,
                total_target_cell_lines,
                cell_line_id,
            )
            break

        output_path = _cell_line_output_path(args.output_dir, cell_line_id)
        output_exists = output_path.exists()
        if output_exists and not args.overwrite_existing:
            skipped_existing += 1
            log.info(
                "Cell line %d/%d: %s (skipped existing %s)",
                index,
                total_target_cell_lines,
                cell_line_id,
                output_path,
            )
            continue

        action = "overwriting existing output" if output_exists else "processing"
        log.info(
            "Cell line %d/%d: %s (%s)",
            index,
            total_target_cell_lines,
            cell_line_id,
            action,
        )

        active_cell_lines = {cell_line_id}
        expression_load_kwargs = _build_expression_stream_load_kwargs(
            hf_load_kwargs,
            cell_lines_whitelist=active_cell_lines,
        )
        _log_expression_stream_load_plan(expression_load_kwargs, active_cell_lines)

        progress_total_cells, progress_total_is_estimated = _resolve_progress_total_cells(
            remaining_sample_size=remaining_sample_size,
            progress_total_cells=args.progress_total_cells,
            target_cell_line_count=total_target_cell_lines,
        )
        stream_open_t0 = time.monotonic()
        stream_name = f"{DATASET_NAME} train split (cell_line_id={cell_line_id})"
        log.info("Opening filtered streaming train split from %s ...", DATASET_NAME)
        raw_records = load_dataset(
            DATASET_NAME,
            streaming=True,
            split="train",
            **expression_load_kwargs,
        )
        log.info(
            "Streaming train split initialized in %.1fs; waiting for first record ...",
            time.monotonic() - stream_open_t0,
        )
        raw_records = _log_first_record_latency(
            raw_records,
            started_at=stream_open_t0,
            stream_name=stream_name,
        )

        progress_reporter = _build_progress_reporter(
            progress=args.progress,
            progress_every=args.progress_every,
            total_cells=progress_total_cells,
            total_is_estimated=progress_total_is_estimated,
            stream=sys.stderr,
        )
        log.info(
            "%s (%s, %s)",
            PROGRESS_PHASE_LABEL,
            _describe_progress_mode(progress_reporter),
            _describe_progress_total(progress_total_cells, progress_total_is_estimated),
        )

        ordered_stream = _iter_ordered_densified_records(
            raw_records,
            n_genes=n_genes,
            token_to_col=token_to_col,
            sample_metadata_by_sample=sample_metadata_by_sample,
            sample_size=remaining_sample_size,
            cell_lines_whitelist=active_cell_lines,
            num_workers=args.num_workers,
        )
        matrix_acc = run_pseudobulk(
            ordered_stream,
            n_genes=n_genes,
            block_size=args.block_size,
            progress_every=args.progress_every,
            progress_reporter=progress_reporter,
        )
        summary = _summarize_accumulators(matrix_acc)
        total_cells_processed += summary.total_cells_processed
        total_cells_used += summary.total_cells_used
        total_cells_dropped += summary.total_cells_dropped
        total_groups_seen += summary.total_groups
        total_groups_with_blocks += summary.groups_with_blocks
        total_groups_without_blocks += summary.groups_without_blocks
        if remaining_sample_size is not None:
            remaining_sample_size = max(
                remaining_sample_size - summary.total_cells_processed,
                0,
            )

        path = write_cell_line_output(
            cell_line_id,
            matrix_acc,
            gene_ids,
            args.output_dir,
            run_meta,
        )
        if path is None:
            skipped_sparse_or_empty += 1
        else:
            written.append(path)

        matrix_acc.cleanup()
        del matrix_acc
        gc.collect()

    if not written and skipped_existing == 0:
        raise RuntimeError(
            "No cell lines produced any complete replicate-aware pseudobulk blocks "
            f"with block size {args.block_size}."
        )

    # ---- Summary ----
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Target cell lines       : %d", total_target_cell_lines)
    log.info("  Total cells processed   : %d", total_cells_processed)
    log.info("  Cells used in blocks    : %d", total_cells_used)
    log.info("  Cells dropped (remainder): %d", total_cells_dropped)
    log.info("  Unique replicate-aware groups: %d", total_groups_seen)
    log.info("    with ≥1 block         : %d", total_groups_with_blocks)
    log.info("    excluded (0 blocks)   : %d", total_groups_without_blocks)
    log.info("  Cell lines written      : %d", len(written))
    log.info("  Cell lines skipped existing: %d", skipped_existing)
    log.info("  Cell lines skipped sparse/empty: %d", skipped_sparse_or_empty)
    log.info(
        "  Stopped early via --sample-size: %s",
        "yes" if stopped_early_due_to_sample_size else "no",
    )
    log.info("  Output directory        : %s", args.output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
