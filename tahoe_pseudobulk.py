"""Tahoe-100M pseudobulk generation.

This script streams Tahoe-100M from Hugging Face, groups records by
cell line and drug treatment, pseudobulks each group, and writes a
cell-line-by-drug cell-count matrix for tracking.

The implementation is designed to run under SLURM:
- streaming dataset access
- bounded worker count
- batch-based threaded preprocessing
- structured logging
- small test-run mode for fast validation
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import gzip
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, DefaultDict, Iterable, Iterator

import anndata as ad
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import sparse


DEFAULT_DATASET_NAME = "vevotx/Tahoe-100M"
DEFAULT_BATCH_SIZE = 512
DEFAULT_REPORT_EVERY = 25_000
DEFAULT_LOOKUP_BATCH_SIZE = 500
DEFAULT_LOOKUP_PAUSE_SECONDS = 0.05
DEFAULT_LOOKUP_TIMEOUT_SECONDS = 30
UNTREATED_LABELS = {"untreated", "control", "vehicle", "dmso"}


@dataclass
class PseudobulkGroup:
    sum_vector: np.ndarray
    n_cells: int
    first_obs: dict


def group_mean_vector(group: PseudobulkGroup) -> np.ndarray:
    if group.n_cells <= 0:
        return np.zeros_like(group.sum_vector)
    return (group.sum_vector / float(group.n_cells)).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Tahoe-100M pseudobulk aggregates.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Number of worker threads used for batch preprocessing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of streamed records per preprocessing batch.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional cap on the number of streamed records.",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Restrict to a small number of treatments for quick validation.",
    )
    parser.add_argument(
        "--test-run-treatments",
        type=int,
        default=8,
        help="Number of unique drugs to retain when --test-run is enabled.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=DEFAULT_REPORT_EVERY,
        help="Log progress every N streamed records.",
    )
    parser.add_argument(
        "--gene-metadata-name",
        default="gene_metadata",
        help="Dataset config name containing gene metadata.",
    )
    parser.add_argument(
        "--sample-metadata-name",
        default="sample_metadata",
        help="Dataset config name containing sample metadata.",
    )
    parser.add_argument(
        "--skip-ensembl-lookup",
        action="store_true",
        help="Skip external Ensembl lookup and keep the raw gene metadata vocabulary.",
    )
    parser.add_argument(
        "--no-filter-nonzero",
        action="store_true",
        help="Disable the strict globally-nonzero gene filter.",
    )
    parser.add_argument(
        "--save-per-cell-line-h5ad",
        "--save-per-group-h5ad",
        dest="save_per_cell_line_h5ad",
        action="store_true",
        help="Write one h5ad file per cell line in addition to the summary CSVs.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional hard cap on the number of cell-line/drug groups processed.",
    )
    parser.add_argument(
        "--checkpoint-every-merges",
        type=int,
        default=25,
        help="Write progress CSV checkpoints every N merged worker batches.",
    )
    parser.add_argument(
        "--save-global-h5ad",
        action="store_true",
        help="Write a single global AnnData object containing all cell-line/drug pseudobulks.",
    )
    parser.add_argument(
        "--save-cell-line-collection-pkl",
        action="store_true",
        help="Write one compressed file containing the full dict of per-cell-line AnnData objects.",
    )
    return parser.parse_args()


def check_file_exists_and_fail(file_path: Path, operation_name: str = "output") -> None:
    """Check if a file already exists and exit if it does to prevent overwriting."""
    if file_path.exists():
        print(f"ERROR: {operation_name} file already exists at {file_path}", file=sys.stderr)
        print(f"To avoid overwriting, please provide a different --output-dir.", file=sys.stderr)
        raise FileExistsError(f"{operation_name} file already exists: {file_path}")


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("tahoe_pseudobulk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "tahoe_pseudobulk.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def fetch_ensembl_biotypes(
    ensembl_ids: Iterable[str],
    *,
    batch_size: int = DEFAULT_LOOKUP_BATCH_SIZE,
    pause_seconds: float = DEFAULT_LOOKUP_PAUSE_SECONDS,
    timeout_seconds: int = DEFAULT_LOOKUP_TIMEOUT_SECONDS,
) -> tuple[dict[str, str | None], int]:
    import requests

    unique_ids = sorted({str(gene_id).split(".")[0] for gene_id in ensembl_ids if gene_id})
    lookup_url = "https://rest.ensembl.org/lookup/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    biotype_by_id: dict[str, str | None] = {}
    failed_batches = 0

    for batch_start in range(0, len(unique_ids), batch_size):
        batch_ids = unique_ids[batch_start : batch_start + batch_size]
        payload = {"ids": batch_ids}
        try:
            response = requests.post(
                lookup_url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            response_payload = response.json()
            for gene_id in batch_ids:
                annotation = response_payload.get(gene_id)
                biotype_by_id[gene_id] = annotation.get("biotype") if isinstance(annotation, dict) else None
        except requests.RequestException:
            failed_batches += 1
            for gene_id in batch_ids:
                biotype_by_id.setdefault(gene_id, None)

        if pause_seconds:
            time.sleep(pause_seconds)

    return biotype_by_id, failed_batches


def build_filtered_gene_vocab(gene_metadata: pd.DataFrame, *, skip_lookup: bool = False) -> tuple[dict[int, str], dict[str, int]]:
    required_columns = {"token_id", "ensembl_id"}
    missing_columns = required_columns - set(gene_metadata.columns)
    if missing_columns:
        raise ValueError(f"Missing required gene metadata columns: {sorted(missing_columns)}")

    gene_df = gene_metadata.loc[:, ["token_id", "ensembl_id"]].dropna().copy()
    initial_genes = int(len(gene_df))

    gene_df = gene_df.drop_duplicates(subset="ensembl_id", keep="first").copy()
    after_dedup_genes = int(len(gene_df))

    gene_df["ensembl_core"] = gene_df["ensembl_id"].astype(str).str.split(".").str[0]

    if skip_lookup:
        gene_df["biotype"] = None
        failed_batches = 0
    else:
        biotype_by_id, failed_batches = fetch_ensembl_biotypes(gene_df["ensembl_core"].tolist())
        gene_df["biotype"] = gene_df["ensembl_core"].map(biotype_by_id)

    resolved_biotypes = int(gene_df["biotype"].notna().sum())
    if not skip_lookup and resolved_biotypes == 0:
        raise RuntimeError(
            "External Ensembl lookup returned no biotype annotations. "
            "Cannot remove pseudogenes without external annotation."
        )

    if skip_lookup:
        pseudogene_mask = pd.Series(False, index=gene_df.index)
    else:
        pseudogene_mask = gene_df["biotype"].fillna("").str.contains("pseudogene", case=False)

    pseudogenes_removed = int(pseudogene_mask.sum())
    gene_df = gene_df.loc[~pseudogene_mask].copy()
    after_pseudogene_genes = int(len(gene_df))

    duplicated_token_ids = int(gene_df["token_id"].duplicated().sum())
    if duplicated_token_ids:
        gene_df = gene_df.drop_duplicates(subset="token_id", keep="first").copy()

    gene_vocab = {int(token_id): ensembl_id for token_id, ensembl_id in zip(gene_df["token_id"], gene_df["ensembl_id"])}

    stats = {
        "initial_genes": initial_genes,
        "after_dedup_genes": after_dedup_genes,
        "pseudogenes_removed": pseudogenes_removed,
        "after_pseudogene_genes": after_pseudogene_genes,
        "resolved_biotypes": resolved_biotypes,
        "unresolved_biotypes": int(after_dedup_genes - resolved_biotypes),
        "failed_lookup_batches": failed_batches,
        "duplicated_token_ids_removed": duplicated_token_ids,
    }

    return gene_vocab, stats


def get_obs_row(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in {"genes", "expressions"}}


def canonicalize_drug_label(drug: object) -> str:
    if drug is None:
        return "Untreated"
    if isinstance(drug, float) and np.isnan(drug):
        return "Untreated"

    label = str(drug).strip()
    if not label:
        return "Untreated"
    if label.lower() in UNTREATED_LABELS:
        return "Untreated"
    return label


def sort_drugs_with_untreated_first(drugs: Iterable[str]) -> list[str]:
    unique_drugs = sorted({canonicalize_drug_label(drug) for drug in drugs})
    if "Untreated" in unique_drugs:
        unique_drugs.remove("Untreated")
        return ["Untreated", *unique_drugs]
    return unique_drugs


def aggregate_record_batch(
    batch: list[dict],
    token_id_to_col_idx: dict[int, int],
    n_genes: int,
    allowed_drugs: set[str] | None,
) -> dict[str, PseudobulkGroup]:
    partial_groups: dict[str, PseudobulkGroup] = {}

    for record in batch:
        drug = canonicalize_drug_label(record["drug"])
        if allowed_drugs is not None and drug not in allowed_drugs:
            continue

        genes = record["genes"]
        expressions = record["expressions"]

        if expressions and expressions[0] < 0:
            genes = genes[1:]
            expressions = expressions[1:]

        group_key = f"{record['cell_line_id']}\t{drug}"
        group = partial_groups.get(group_key)
        if group is None:
            group = PseudobulkGroup(
                sum_vector=np.zeros(n_genes, dtype=np.float32),
                n_cells=0,
                first_obs=get_obs_row(record),
            )
            partial_groups[group_key] = group

        for gene_token, expr_value in zip(genes, expressions):
            col_idx = token_id_to_col_idx.get(gene_token)
            if col_idx is not None:
                group.sum_vector[col_idx] += float(expr_value)

        group.n_cells += 1

    return partial_groups


def iter_record_batches(streaming_ds, *, batch_size: int, sample_size: int | None) -> Iterator[tuple[int, list[dict]]]:
    batch: list[dict] = []
    for index, record in enumerate(streaming_ds):
        if sample_size is not None and index >= sample_size:
            break
        batch.append(record)
        if len(batch) >= batch_size:
            yield index + 1, batch
            batch = []

    if batch:
        yield index + 1, batch


def build_pseudobulk_index(
    streaming_ds,
    gene_vocab: dict[int, str],
    *,
    num_workers: int,
    batch_size: int,
    sample_size: int | None,
    report_every: int,
    logger: logging.Logger,
    allowed_drugs: set[str] | None = None,
    max_groups: int | None = None,
    checkpoint_every_merges: int = 25,
    checkpoint_callback: Callable[[DefaultDict[str, PseudobulkGroup], int], None] | None = None,
) -> tuple[dict[str, dict[str, PseudobulkGroup]], list[str]]:
    sorted_vocab_items = sorted(gene_vocab.items())
    token_ids, gene_names = zip(*sorted_vocab_items) if sorted_vocab_items else ([], [])
    token_id_to_col_idx = {token_id: idx for idx, token_id in enumerate(token_ids)}

    grouped_buffers: DefaultDict[str, PseudobulkGroup] = defaultdict()
    ordered_groups: list[str] = []

    total_streamed = 0
    merged_batches = 0

    def submit_batch(executor, batch_records: list[dict]):
        if not batch_records:
            return None
        return executor.submit(
            aggregate_record_batch,
            batch_records,
            token_id_to_col_idx,
            len(gene_names),
            allowed_drugs,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        pending: list[concurrent.futures.Future[dict[str, PseudobulkGroup]]] = []

        for streamed_count, batch_records in iter_record_batches(
            streaming_ds,
            batch_size=batch_size,
            sample_size=sample_size,
        ):
            total_streamed = streamed_count
            if batch_records:
                pending.append(submit_batch(executor, batch_records))
            if total_streamed and report_every > 0 and total_streamed % report_every == 0:
                logger.info("Streamed %s records", total_streamed)

            if len(pending) >= max(2, num_workers * 2):
                done, pending = wait_for_some(pending)
                for future in done:
                    partial_groups = future.result()
                    merge_partial_groups(
                        partial_groups,
                        grouped_buffers,
                        ordered_groups,
                        logger,
                        max_groups=max_groups,
                    )
                    merged_batches += 1
                    if (
                        checkpoint_callback is not None
                        and checkpoint_every_merges > 0
                        and merged_batches % checkpoint_every_merges == 0
                    ):
                        checkpoint_callback(grouped_buffers, total_streamed)

        for future in concurrent.futures.as_completed(pending):
            partial_groups = future.result()
            merge_partial_groups(
                partial_groups,
                grouped_buffers,
                ordered_groups,
                logger,
                max_groups=max_groups,
            )
            merged_batches += 1
            if (
                checkpoint_callback is not None
                and checkpoint_every_merges > 0
                and merged_batches % checkpoint_every_merges == 0
            ):
                checkpoint_callback(grouped_buffers, total_streamed)

    if checkpoint_callback is not None:
        checkpoint_callback(grouped_buffers, total_streamed)

    data_by_cell_and_drug: dict[str, dict[str, PseudobulkGroup]] = defaultdict(dict)
    for group_key in ordered_groups:
        cell_line_id, perturbation = group_key.split("\t", 1)
        data_by_cell_and_drug[cell_line_id][perturbation] = grouped_buffers[group_key]

    return data_by_cell_and_drug, list(gene_names)


def wait_for_some(
    pending: list[concurrent.futures.Future[dict[str, PseudobulkGroup]]],
) -> tuple[
    list[concurrent.futures.Future[dict[str, PseudobulkGroup]]],
    list[concurrent.futures.Future[dict[str, PseudobulkGroup]]],
]:
    done: list[concurrent.futures.Future[dict[str, PseudobulkGroup]]] = []
    still_pending: list[concurrent.futures.Future[dict[str, PseudobulkGroup]]] = []
    for future in pending:
        if future.done():
            done.append(future)
        else:
            still_pending.append(future)
    if done:
        return done, still_pending

    first = pending[0]
    done_future = next(concurrent.futures.as_completed([first]))
    done = [done_future]
    still_pending = [future for future in pending if future is not done_future]
    return done, still_pending


def merge_partial_groups(
    partial_groups: dict[str, PseudobulkGroup],
    grouped_buffers: DefaultDict[str, PseudobulkGroup],
    ordered_groups: list[str],
    logger: logging.Logger,
    *,
    max_groups: int | None,
) -> None:
    for group_key, partial in partial_groups.items():
        if max_groups is not None and group_key not in grouped_buffers and len(grouped_buffers) >= max_groups:
            continue

        if group_key not in grouped_buffers:
            grouped_buffers[group_key] = PseudobulkGroup(
                sum_vector=partial.sum_vector.copy(),
                n_cells=partial.n_cells,
                first_obs=partial.first_obs,
            )
            ordered_groups.append(group_key)
            cell_line_id, drug = group_key.split("\t", 1)
            logger.info("Started group cell_line=%s drug=%s", cell_line_id, drug)
        else:
            grouped_buffers[group_key].sum_vector += partial.sum_vector
            grouped_buffers[group_key].n_cells += partial.n_cells

        cell_line_id, drug = group_key.split("\t", 1)
        logger.debug("Merged group cell_line=%s drug=%s cells=%s", cell_line_id, drug, partial.n_cells)


def filter_to_globally_nonzero_genes(
    data_by_cell_and_drug: dict[str, dict[str, PseudobulkGroup]],
    gene_names: list[str],
) -> tuple[dict[str, dict[str, PseudobulkGroup]], list[str], dict[str, int]]:
    if not gene_names:
        return data_by_cell_and_drug, gene_names, {
            "total_cells": 0,
            "input_genes": 0,
            "genes_with_any_zero": 0,
            "kept_genes": 0,
        }

    total_cells = 0
    nonzero_counts = np.zeros(len(gene_names), dtype=np.int64)

    for perturbation_map in data_by_cell_and_drug.values():
        for group in perturbation_map.values():
            nonzero_counts += (group.sum_vector > 0).astype(np.int64, copy=False)
            total_cells += 1

    keep_mask = nonzero_counts == total_cells
    kept_gene_names = [gene_name for gene_name, keep in zip(gene_names, keep_mask) if keep]

    for cell_line_id, perturbation_map in data_by_cell_and_drug.items():
        for perturbation, group in perturbation_map.items():
            group.sum_vector = group.sum_vector[keep_mask].copy()

    stats = {
        "total_cells": int(total_cells),
        "input_genes": int(len(gene_names)),
        "genes_with_any_zero": int((~keep_mask).sum()),
        "kept_genes": int(keep_mask.sum()),
    }

    return data_by_cell_and_drug, kept_gene_names, stats


def summarize_groups(data_by_cell_and_drug: dict[str, dict[str, PseudobulkGroup]]) -> pd.DataFrame:
    rows = []
    for cell_line_id, perturbation_map in data_by_cell_and_drug.items():
        for perturbation, group in perturbation_map.items():
            rows.append(
                {
                    "cell_line_id": cell_line_id,
                    "drug": perturbation,
                    "n_cells": int(group.n_cells),
                    "n_genes": int(group.sum_vector.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def count_cells_matrix(data_by_cell_and_drug: dict[str, dict[str, PseudobulkGroup]]) -> pd.DataFrame:
    records = []
    for cell_line_id, perturbation_map in data_by_cell_and_drug.items():
        for drug, group in perturbation_map.items():
            records.append(
                {
                    "cell_line_id": cell_line_id,
                    "drug": drug,
                    "n_cells": int(group.n_cells),
                }
            )
    if not records:
        return pd.DataFrame(columns=["cell_line_id", "drug", "n_cells"])
    return pd.DataFrame(records)


def build_coverage_report(
    sample_metadata: pd.DataFrame,
    count_df: pd.DataFrame,
) -> pd.DataFrame:
    observed_cell_lines = int(count_df["cell_line_id"].nunique()) if not count_df.empty else 0
    observed_drugs = int(count_df["drug"].nunique()) if not count_df.empty else 0
    observed_pairs = int(len(count_df))

    metadata_cell_lines = int(sample_metadata["cell_line_id"].astype(str).nunique()) if "cell_line_id" in sample_metadata.columns else observed_cell_lines
    metadata_drugs = (
        int(sample_metadata["drug"].map(canonicalize_drug_label).nunique())
        if "drug" in sample_metadata.columns
        else observed_drugs
    )

    possible_pairs = int(metadata_cell_lines * metadata_drugs)
    coverage_fraction = float(observed_pairs / possible_pairs) if possible_pairs else 0.0
    missing_pairs = int(max(possible_pairs - observed_pairs, 0))

    return pd.DataFrame(
        [
            {
                "metadata_cell_lines": metadata_cell_lines,
                "metadata_drugs": metadata_drugs,
                "possible_pairs": possible_pairs,
                "observed_cell_lines": observed_cell_lines,
                "observed_drugs": observed_drugs,
                "observed_pairs": observed_pairs,
                "missing_pairs": missing_pairs,
                "coverage_fraction": coverage_fraction,
            }
        ]
    )


def pseudobulk_df_for_pair(
    data_by_cell_and_drug: dict[str, dict[str, PseudobulkGroup]],
    cell_line_id: str,
    drug: str,
    gene_names: list[str],
) -> pd.DataFrame:
    group = data_by_cell_and_drug[cell_line_id][drug]
    row = pd.DataFrame([group.sum_vector], columns=gene_names)
    row.insert(0, "cell_line_id", cell_line_id)
    row.insert(1, "drug", drug)
    row.insert(2, "n_cells", group.n_cells)
    return row


def build_treatment_subset(sample_metadata: pd.DataFrame, test_run: bool, test_run_treatments: int) -> set[str]:
    if "drug" not in sample_metadata.columns:
        raise ValueError("Sample metadata is missing the 'drug' column.")

    all_perturbations = sort_drugs_with_untreated_first(sample_metadata["drug"].dropna().tolist())
    if not test_run:
        return set(all_perturbations)

    rng = np.random.default_rng(42)
    fixed = [drug for drug in all_perturbations if drug == "Untreated"]
    variable = np.array([drug for drug in all_perturbations if drug != "Untreated"], dtype=object)
    rng.shuffle(variable)

    selected = fixed[:]
    remaining_slots = max(test_run_treatments - len(selected), 0)
    selected.extend(variable[:remaining_slots].tolist())
    return set(selected)


def build_cell_line_adata_collection(
    data_by_cell_and_drug: dict[str, dict[str, PseudobulkGroup]],
    gene_names: list[str],
    cell_line_order: list[str],
    drug_order: list[str],
) -> tuple[dict[str, ad.AnnData], pd.DataFrame]:
    cell_line_to_adata: dict[str, ad.AnnData] = {}
    manifest_rows = []
    gene_index = pd.Index(gene_names, name="ensembl_id")

    for cell_line_id in cell_line_order:
        perturbation_map = data_by_cell_and_drug.get(cell_line_id, {})
        rows = []
        n_cells = []
        observed = []
        for drug in drug_order:
            group = perturbation_map.get(drug)
            if group is None:
                rows.append(np.zeros(len(gene_names), dtype=np.float32))
                n_cells.append(0)
                observed.append(False)
            else:
                rows.append(group_mean_vector(group))
                n_cells.append(int(group.n_cells))
                observed.append(True)

        obs = pd.DataFrame(
            {
                "drug": drug_order,
                "n_cells": n_cells,
                "observed": observed,
            },
            index=pd.Index(drug_order, name="drug"),
        )

        adata = ad.AnnData(X=np.vstack(rows).astype(np.float32, copy=False), obs=obs)
        adata.var.index = gene_index
        adata.uns["cell_line_id"] = cell_line_id
        adata.uns["row_order"] = drug_order

        cell_line_to_adata[cell_line_id] = adata
        manifest_rows.append(
            {
                "cell_line_id": cell_line_id,
                "n_drugs": int(len(drug_order)),
                "n_genes": int(len(gene_names)),
                "untreated_present": bool("Untreated" in perturbation_map),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    return cell_line_to_adata, manifest_df


def write_cell_line_outputs(
    cell_line_to_adata: dict[str, ad.AnnData],
    output_dir: Path,
) -> pd.DataFrame:
    cell_line_dir = output_dir / "cell_line_pseudobulk_h5ad"
    check_file_exists_and_fail(cell_line_dir, "cell_line_pseudobulk_h5ad directory")
    cell_line_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for cell_line_id, adata in sorted(cell_line_to_adata.items()):
        safe_name = f"cellline={cell_line_id}".replace("/", "_").replace(" ", "_")
        file_path = cell_line_dir / f"{safe_name}.h5ad"
        if file_path.exists():
            raise FileExistsError(f"h5ad file already exists for cell line {cell_line_id}: {file_path}")
        adata.write_h5ad(file_path)
        manifest_rows.append(
            {
                "cell_line_id": cell_line_id,
                "path": str(file_path),
                "n_drugs": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "untreated_first": bool(len(adata.obs.index) > 0 and adata.obs.index[0] == "Untreated"),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / "cell_line_pseudobulk_manifest.csv", index=False)
    return manifest_df


def write_cell_line_collection_pickle(
    cell_line_to_adata: dict[str, ad.AnnData],
    output_dir: Path,
) -> Path:
    collection_path = output_dir / "cell_line_adata_collection.pkl.gz"
    check_file_exists_and_fail(collection_path, "cell_line_adata_collection.pkl.gz")
    payload = {
        "format": "cell_line_to_anndata_dict_v1",
        "cell_lines": sorted(cell_line_to_adata.keys()),
        "data": cell_line_to_adata,
    }
    with gzip.open(collection_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return collection_path


def build_global_adata(cell_line_to_adata: dict[str, ad.AnnData], gene_names: list[str]) -> ad.AnnData:
    rows = []
    obs_rows = []

    for cell_line_id, adata in sorted(cell_line_to_adata.items()):
        x_matrix = adata.X
        x_sparse = sparse.csr_matrix(x_matrix)
        rows.append(x_sparse)

        for idx, obs_row in adata.obs.iterrows():
            obs_rows.append(
                {
                    "cell_line_id": cell_line_id,
                    "drug": str(obs_row["drug"]),
                    "n_cells": int(obs_row["n_cells"]),
                    "observed": bool(obs_row["observed"]),
                    "pair_id": f"{cell_line_id}__{idx}",
                }
            )

    if rows:
        x_global = sparse.vstack(rows, format="csr", dtype=np.float32)
    else:
        x_global = sparse.csr_matrix((0, len(gene_names)), dtype=np.float32)

    obs_df = pd.DataFrame(obs_rows)
    if not obs_df.empty:
        obs_df = obs_df.set_index("pair_id", drop=True)
    else:
        obs_df = pd.DataFrame(columns=["cell_line_id", "drug", "n_cells", "observed"])

    global_adata = ad.AnnData(X=x_global, obs=obs_df)
    global_adata.var.index = pd.Index(gene_names, name="ensembl_id")
    global_adata.uns["layout"] = "rows are cell_line_id/drug pairs; columns are genes"
    return global_adata


def atomic_write_csv(df: pd.DataFrame, output_path: Path, *, index: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    df.to_csv(temp_path, index=index)
    os.replace(temp_path, output_path)


def grouped_buffers_to_count_df(grouped_buffers: DefaultDict[str, PseudobulkGroup]) -> pd.DataFrame:
    records = []
    for group_key, group in grouped_buffers.items():
        cell_line_id, drug = group_key.split("\t", 1)
        records.append(
            {
                "cell_line_id": cell_line_id,
                "drug": drug,
                "n_cells": int(group.n_cells),
            }
        )
    if not records:
        return pd.DataFrame(columns=["cell_line_id", "drug", "n_cells"])
    return pd.DataFrame(records)


def write_progress_checkpoints(
    grouped_buffers: DefaultDict[str, PseudobulkGroup],
    streamed_records: int,
    output_dir: Path,
    *,
    cell_line_order: list[str],
    drug_order: list[str],
) -> None:
    progress_dir = output_dir / "progress"
    count_df = grouped_buffers_to_count_df(grouped_buffers)

    atomic_write_csv(
        count_df,
        progress_dir / "cell_line_drug_cell_counts_long.csv",
        index=False,
    )

    matrix = count_df.pivot_table(
        index="cell_line_id",
        columns="drug",
        values="n_cells",
        fill_value=0,
        aggfunc="sum",
    ) if not count_df.empty else pd.DataFrame()
    matrix = matrix.reindex(index=cell_line_order, fill_value=0)
    matrix = matrix.reindex(columns=drug_order, fill_value=0)
    atomic_write_csv(
        matrix,
        progress_dir / "cell_line_drug_cell_counts.csv",
        index=True,
    )

    status_df = pd.DataFrame(
        [
            {
                "streamed_records": int(streamed_records),
                "observed_groups": int(len(grouped_buffers)),
                "checkpoint_unix_time": float(time.time()),
            }
        ]
    )
    atomic_write_csv(status_df, progress_dir / "status.csv", index=False)


def write_count_matrix(
    count_df: pd.DataFrame,
    output_dir: Path,
    *,
    cell_line_order: list[str] | None = None,
    drug_order: list[str] | None = None,
) -> None:
    output_path = output_dir / "cell_line_drug_cell_counts.csv"
    if count_df.empty:
        empty_matrix = pd.DataFrame(index=cell_line_order or [], columns=drug_order or []).fillna(0)
        empty_matrix.to_csv(output_path)
        return

    matrix = count_df.pivot_table(index="cell_line_id", columns="drug", values="n_cells", fill_value=0, aggfunc="sum")
    if cell_line_order is not None:
        matrix = matrix.reindex(index=cell_line_order, fill_value=0)
    if drug_order is not None:
        matrix = matrix.reindex(columns=drug_order, fill_value=0)
    if cell_line_order is None:
        matrix.sort_index(axis=0, inplace=True)
    if drug_order is None:
        matrix.sort_index(axis=1, inplace=True)
    matrix.to_csv(output_path)


def main() -> int:
    args = parse_args()
    
    # Create outputs folder with timestamp to organize runs
    base_output_dir = Path(args.output_dir).expanduser().resolve() / "outputs"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base_output_dir / timestamp
    
    # Check if the timestamped directory already exists (should not happen in normal usage)
    if output_dir.exists():
        print(f"ERROR: Output directory already exists: {output_dir}", file=sys.stderr)
        print(f"This is unexpected as timestamps should be unique. Please check for clock issues.", file=sys.stderr)
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    
    logger = configure_logging(output_dir)

    logger.info("Starting Tahoe-100M pseudobulk generation")
    logger.info("Dataset=%s split=%s workers=%s batch_size=%s sample_size=%s test_run=%s", args.dataset_name, args.split, args.num_workers, args.batch_size, args.sample_size, args.test_run)
    logger.info("SLURM job id=%s node=%s", os.environ.get("SLURM_JOB_ID", "n/a"), os.environ.get("SLURM_NODELIST", "n/a"))

    sample_metadata = load_dataset(args.dataset_name, name=args.sample_metadata_name, split=args.split)
    gene_metadata = load_dataset(args.dataset_name, name=args.gene_metadata_name, split=args.split)

    if hasattr(gene_metadata, "to_pandas"):
        gene_metadata_df = gene_metadata.to_pandas()
    else:
        gene_metadata_df = pd.DataFrame(gene_metadata)

    gene_vocab, gene_filter_stats = build_filtered_gene_vocab(
        gene_metadata_df,
        skip_lookup=args.skip_ensembl_lookup,
    )
    logger.info("Gene vocab stats: %s", gene_filter_stats)

    sample_metadata_df = sample_metadata.to_pandas() if hasattr(sample_metadata, "to_pandas") else pd.DataFrame(sample_metadata)

    treatment_subset = build_treatment_subset(
        sample_metadata_df,
        args.test_run,
        args.test_run_treatments,
    )
    logger.info("Selected %s treatments", len(treatment_subset))

    streaming_ds = load_dataset(args.dataset_name, streaming=True, split=args.split)

    all_cell_lines = (
        sorted(sample_metadata_df["cell_line_id"].dropna().astype(str).unique().tolist())
        if "cell_line_id" in sample_metadata_df.columns
        else []
    )
    all_drugs = (
        sort_drugs_with_untreated_first(sample_metadata_df["drug"].dropna().tolist())
        if "drug" in sample_metadata_df.columns
        else sort_drugs_with_untreated_first(treatment_subset)
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_callback(grouped_buffers: DefaultDict[str, PseudobulkGroup], streamed_records: int) -> None:
        write_progress_checkpoints(
            grouped_buffers,
            streamed_records,
            output_dir,
            cell_line_order=all_cell_lines,
            drug_order=all_drugs,
        )

    data_by_cell_and_drug, gene_names = build_pseudobulk_index(
        streaming_ds,
        gene_vocab,
        num_workers=max(1, args.num_workers),
        batch_size=max(1, args.batch_size),
        sample_size=args.sample_size,
        report_every=max(1, args.report_every),
        logger=logger,
        allowed_drugs=treatment_subset,
        max_groups=args.max_groups,
        checkpoint_every_merges=max(1, args.checkpoint_every_merges),
        checkpoint_callback=checkpoint_callback,
    )

    if not args.no_filter_nonzero:
        data_by_cell_and_drug, gene_names, nonzero_stats = filter_to_globally_nonzero_genes(data_by_cell_and_drug, gene_names)
        logger.info("Nonzero gene filter stats: %s", nonzero_stats)
    else:
        logger.info("Skipped global nonzero gene filtering")

    group_summary_df = summarize_groups(data_by_cell_and_drug)
    count_df = count_cells_matrix(data_by_cell_and_drug)
    coverage_df = build_coverage_report(sample_metadata_df, count_df)

    if not all_cell_lines:
        all_cell_lines = sorted(group_summary_df["cell_line_id"].dropna().astype(str).unique().tolist())
    if not all_drugs:
        all_drugs = sort_drugs_with_untreated_first(group_summary_df["drug"].dropna().tolist())

    cell_line_to_adata, cell_line_manifest_df = build_cell_line_adata_collection(
        data_by_cell_and_drug,
        gene_names,
        all_cell_lines,
        all_drugs,
    )
    logger.info("Built %s cell-line AnnData objects", len(cell_line_to_adata))

    # Check key output files don't already exist
    output_dir.mkdir(parents=True, exist_ok=True)
    check_file_exists_and_fail(output_dir / "group_summary.csv", "group_summary.csv")
    check_file_exists_and_fail(output_dir / "cell_line_drug_cell_counts_long.csv", "cell_line_drug_cell_counts_long.csv")
    check_file_exists_and_fail(output_dir / "cell_line_drug_cell_counts.csv", "cell_line_drug_cell_counts.csv")

    group_summary_df.to_csv(output_dir / "group_summary.csv", index=False)
    logger.info("Wrote group_summary.csv to %s", output_dir / "group_summary.csv")
    
    count_df.to_csv(output_dir / "cell_line_drug_cell_counts_long.csv", index=False)
    logger.info("Wrote cell_line_drug_cell_counts_long.csv to %s", output_dir / "cell_line_drug_cell_counts_long.csv")
    
    write_count_matrix(count_df, output_dir, cell_line_order=all_cell_lines, drug_order=all_drugs)
    logger.info("Wrote cell_line_drug_cell_counts.csv to %s", output_dir / "cell_line_drug_cell_counts.csv")
    
    coverage_df.to_csv(output_dir / "coverage_report.csv", index=False)
    logger.info("Wrote coverage_report.csv to %s", output_dir / "coverage_report.csv")
    
    cell_line_manifest_df.to_csv(output_dir / "cell_line_collection_manifest.csv", index=False)
    logger.info("Wrote cell_line_collection_manifest.csv to %s", output_dir / "cell_line_collection_manifest.csv")

    gene_names_path = output_dir / "gene_names.txt"
    check_file_exists_and_fail(gene_names_path, "gene_names.txt")
    with gene_names_path.open("w", encoding="utf-8") as handle:
        for gene_name in gene_names:
            handle.write(f"{gene_name}\n")
    logger.info("Wrote gene_names.txt to %s", gene_names_path)

    if args.save_per_cell_line_h5ad:
        write_cell_line_outputs(cell_line_to_adata, output_dir)
        logger.info("Wrote per-cell-line h5ad files to %s", output_dir / "cell_line_pseudobulk_h5ad")

    if args.save_cell_line_collection_pkl:
        collection_path = write_cell_line_collection_pickle(cell_line_to_adata, output_dir)
        logger.info("Wrote cell-line AnnData collection to %s", collection_path)

    if args.save_global_h5ad:
        global_adata = build_global_adata(cell_line_to_adata, gene_names)
        global_h5ad_path = output_dir / "all_cell_lines_pseudobulk.h5ad"
        check_file_exists_and_fail(global_h5ad_path, "all_cell_lines_pseudobulk.h5ad")
        global_adata.write_h5ad(global_h5ad_path)
        logger.info("Wrote global AnnData to %s with shape=%s", global_h5ad_path, global_adata.shape)

    if group_summary_df.empty:
        logger.warning("No groups were produced. Check sample size, treatment selection, or dataset access.")
        return 1

    logger.info("Generated %s cell-line/drug groups", len(group_summary_df))
    logger.info("Unique cell lines=%s unique drugs=%s", group_summary_df["cell_line_id"].nunique(), group_summary_df["drug"].nunique())
    logger.info("Coverage: %s observed pairs out of %s possible (%.2f%%)", int(coverage_df.iloc[0]["observed_pairs"]), int(coverage_df.iloc[0]["possible_pairs"]), float(100.0 * coverage_df.iloc[0]["coverage_fraction"]))
    logger.info("Run timestamp: %s", timestamp)
    logger.info("Wrote outputs to %s", output_dir)

    example_cell_line = group_summary_df.iloc[0]["cell_line_id"]
    example_adata = cell_line_to_adata[example_cell_line]
    logger.info("Example cell line %s AnnData shape=%s (drugs x genes)", example_cell_line, example_adata.shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())