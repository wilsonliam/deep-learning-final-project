"""Helpers for Tahoe-100M gene-biotype filtering in notebooks.

These utilities are designed for three common tasks:
1. Build a protein-coding gene list from Tahoe gene metadata.
2. Lazily filter a streaming differential-expression dataset by gene name.
3. Materialize filtered rows to disk in JSONL or sharded Parquet form.

For streaming datasets, filtering remains lazy until you iterate. If you want
to reuse the subset later without re-streaming the full source, materialize the
filtered rows once to disk and reload them with `datasets.load_dataset`.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from tahoe_pseudobulk import fetch_ensembl_biotypes


def load_gene_metadata_frame(
    dataset_name: str = "vevotx/Tahoe-100M",
    *,
    gene_metadata_name: str = "gene_metadata",
    split: str = "train",
) -> pd.DataFrame:
    """Load Tahoe gene metadata into a pandas DataFrame."""
    gene_metadata = load_dataset(dataset_name, name=gene_metadata_name, split=split)
    return pd.DataFrame(gene_metadata)


def resolve_gene_biotypes(
    gene_metadata: pd.DataFrame,
    *,
    use_existing_biotype: bool = True,
    lookup_missing: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach gene biotypes to Tahoe gene metadata.

    The Tahoe `gene_metadata` split may not include a native `biotype` column.
    When absent, this function can resolve Ensembl biotypes using the existing
    lookup helper from `tahoe_pseudobulk.py`.
    """
    required_columns = {"token_id", "ensembl_id"}
    missing_columns = required_columns - set(gene_metadata.columns)
    if missing_columns:
        raise ValueError(f"Missing required gene metadata columns: {sorted(missing_columns)}")

    base_columns = [col for col in ("token_id", "ensembl_id", "gene_symbol") if col in gene_metadata.columns]
    gene_df = gene_metadata.loc[:, base_columns].dropna(subset=["token_id", "ensembl_id"]).copy()
    initial_genes = int(len(gene_df))

    gene_df["ensembl_core"] = gene_df["ensembl_id"].astype(str).str.split(".").str[0]
    gene_df = gene_df.drop_duplicates(subset="ensembl_core", keep="first").copy()
    after_dedup_genes = int(len(gene_df))

    local_biotype_col = None
    if use_existing_biotype:
        local_biotype_col = next(
            (column for column in ("biotype", "gene_biotype") if column in gene_metadata.columns),
            None,
        )

    if local_biotype_col is not None:
        biotype_lookup = (
            gene_metadata.loc[:, ["ensembl_id", local_biotype_col]]
            .dropna(subset=["ensembl_id"])
            .assign(ensembl_core=lambda df: df["ensembl_id"].astype(str).str.split(".").str[0])
            .drop_duplicates(subset="ensembl_core", keep="first")
        )
        gene_df["biotype"] = gene_df["ensembl_core"].map(
            dict(zip(biotype_lookup["ensembl_core"], biotype_lookup[local_biotype_col]))
        )
        failed_lookup_batches = 0
    elif lookup_missing:
        biotype_by_id, failed_lookup_batches = fetch_ensembl_biotypes(gene_df["ensembl_core"].tolist())
        gene_df["biotype"] = gene_df["ensembl_core"].map(biotype_by_id)
    else:
        gene_df["biotype"] = pd.NA
        failed_lookup_batches = 0

    gene_df["is_protein_coding"] = gene_df["biotype"].fillna("").eq("protein_coding")

    stats = {
        "initial_genes": initial_genes,
        "after_dedup_genes": after_dedup_genes,
        "resolved_biotypes": int(gene_df["biotype"].notna().sum()),
        "unresolved_biotypes": int(gene_df["biotype"].isna().sum()),
        "protein_coding_genes": int(gene_df["is_protein_coding"].sum()),
        "failed_lookup_batches": failed_lookup_batches,
    }
    return gene_df, stats


def protein_coding_gene_symbols(gene_biotypes: pd.DataFrame) -> list[str]:
    """Return a sorted list of protein-coding gene symbols."""
    if "gene_symbol" not in gene_biotypes.columns:
        raise ValueError("Expected a 'gene_symbol' column in the resolved gene-biotype table.")

    symbols = (
        gene_biotypes.loc[gene_biotypes["is_protein_coding"], "gene_symbol"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
    )
    return symbols.tolist()


def filter_streaming_de_by_gene_name(streaming_dataset, gene_names: Sequence[str], *, gene_field: str = "gene_name"):
    """Return a lazy filtered streaming DE dataset."""
    gene_name_set = set(gene_names)
    return streaming_dataset.filter(lambda row: row.get(gene_field) in gene_name_set)


def iter_matching_rows(rows: Iterable[dict], gene_names: Sequence[str], *, gene_field: str = "gene_name") -> Iterator[dict]:
    """Yield only rows whose gene field matches the provided gene names."""
    gene_name_set = set(gene_names)
    for row in rows:
        if row.get(gene_field) in gene_name_set:
            yield row


def save_rows_to_jsonl(rows: Iterable[dict], output_path: str | Path, *, limit: int | None = None) -> int:
    """Materialize streamed rows to JSONL for reuse later."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if limit is not None and n_written >= limit:
                break
            handle.write(json.dumps(row, default=_json_default))
            handle.write("\n")
            n_written += 1

    return n_written


def save_rows_to_parquet_shards(
    rows: Iterable[dict],
    output_dir: str | Path,
    *,
    rows_per_file: int = 250_000,
    num_workers: int = 4,
    compression: str = "zstd",
    prefix: str = "part",
    limit: int | None = None,
    max_pending_files: int | None = None,
) -> dict[str, int | str]:
    """Materialize streamed rows to sharded Parquet files.

    Reading from the source iterable remains single-pass. Each completed batch
    is handed to a worker that writes one Parquet shard, which allows the
    serialization and file writes to overlap with ongoing streaming.
    """
    if rows_per_file <= 0:
        raise ValueError("rows_per_file must be a positive integer.")
    if num_workers <= 0:
        raise ValueError("num_workers must be a positive integer.")

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    pending_limit = max_pending_files or max(2, num_workers * 2)
    schema: pa.Schema | None = None
    batch: list[dict] = []
    pending: list[concurrent.futures.Future] = []
    n_rows_written = 0
    n_files_written = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for row in rows:
            if limit is not None and n_rows_written + len(batch) >= limit:
                break
            batch.append(_normalize_row(row))

            if limit is not None and n_rows_written + len(batch) >= limit:
                batch = batch[: limit - n_rows_written]

            if len(batch) < rows_per_file:
                continue

            schema = _submit_parquet_batch(
                executor,
                pending,
                batch,
                path,
                prefix,
                n_files_written,
                compression,
                schema,
            )
            n_rows_written += len(batch)
            n_files_written += 1
            batch = []

            if len(pending) >= pending_limit:
                _drain_completed_writes(pending, wait_for_one=True)

        if batch:
            schema = _submit_parquet_batch(
                executor,
                pending,
                batch,
                path,
                prefix,
                n_files_written,
                compression,
                schema,
            )
            n_rows_written += len(batch)
            n_files_written += 1

        _drain_completed_writes(pending, wait_for_one=False)

    return {
        "rows_written": n_rows_written,
        "files_written": n_files_written,
        "output_dir": str(path),
    }


def save_gene_list(genes: Sequence[str], output_path: str | Path) -> int:
    """Write one gene symbol per line."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_genes = sorted({str(gene) for gene in genes})
    path.write_text("\n".join(unique_genes) + "\n", encoding="utf-8")
    return len(unique_genes)


def _json_default(value):
    if pd.isna(value):
        return None
    return value


def _normalize_row(row: dict) -> dict:
    return {key: _normalize_value(value) for key, value in row.items()}


def _normalize_value(value):
    if isinstance(value, dict):
        return {key: _normalize_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(inner) for inner in value]
    if pd.isna(value):
        return None
    return value


def _submit_parquet_batch(
    executor: concurrent.futures.ThreadPoolExecutor,
    pending: list[concurrent.futures.Future],
    batch: list[dict],
    output_dir: Path,
    prefix: str,
    shard_idx: int,
    compression: str,
    schema: pa.Schema | None,
) -> pa.Schema:
    batch_rows = list(batch)
    if schema is None:
        schema = pa.Table.from_pylist(batch_rows).schema

    output_path = output_dir / f"{prefix}-{shard_idx:05d}.parquet"
    future = executor.submit(_write_parquet_file, batch_rows, output_path, schema, compression)
    pending.append(future)
    return schema


def _write_parquet_file(
    rows: list[dict],
    output_path: Path,
    schema: pa.Schema,
    compression: str,
) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output_path, compression=compression)


def _drain_completed_writes(
    pending: list[concurrent.futures.Future],
    *,
    wait_for_one: bool,
) -> None:
    if not pending:
        return

    if wait_for_one:
        done, not_done = concurrent.futures.wait(
            pending,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for future in done:
            future.result()
        pending[:] = list(not_done)
        return

    for future in concurrent.futures.as_completed(pending):
        future.result()
    pending.clear()
