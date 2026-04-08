"""Load and validate a compressed cell-line AnnData collection.

Expected input is the gzip pickle written by tahoe_pseudobulk.py with the
--save-cell-line-collection-pkl flag.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and validate cell-line AnnData collection.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to cell_line_adata_collection.pkl.gz",
    )
    parser.add_argument(
        "--expected-cell-lines",
        type=int,
        default=50,
        help="Expected number of cell lines in the collection.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if validation checks fail.",
    )
    parser.add_argument(
        "--check-untreated-first",
        action="store_true",
        help="Validate that Untreated is the first row where present.",
    )
    return parser.parse_args()


def load_collection(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Collection payload is not a dictionary.")
    if "data" not in payload:
        raise ValueError("Collection payload is missing 'data'.")
    if not isinstance(payload["data"], dict):
        raise ValueError("Collection payload 'data' is not a dictionary.")

    return payload


def summarize_collection(payload: dict) -> tuple[list[str], int, int]:
    cell_line_to_adata = payload["data"]
    cell_lines = sorted(cell_line_to_adata.keys())

    total_drug_rows = 0
    n_genes = 0
    for adata in cell_line_to_adata.values():
        total_drug_rows += int(adata.n_obs)
        n_genes = max(n_genes, int(adata.n_vars))

    return cell_lines, total_drug_rows, n_genes


def validate_collection(payload: dict, expected_cell_lines: int, check_untreated_first: bool) -> list[str]:
    issues: list[str] = []
    cell_line_to_adata = payload["data"]

    if len(cell_line_to_adata) != expected_cell_lines:
        issues.append(
            f"Expected {expected_cell_lines} cell lines, found {len(cell_line_to_adata)}."
        )

    for cell_line_id, adata in cell_line_to_adata.items():
        if "drug" not in adata.obs.columns:
            issues.append(f"{cell_line_id}: missing obs['drug'] column.")
        if "n_cells" not in adata.obs.columns:
            issues.append(f"{cell_line_id}: missing obs['n_cells'] column.")
        if check_untreated_first and int(adata.n_obs) > 0:
            observed_drugs = [str(value) for value in adata.obs.index.tolist()]
            if "Untreated" in observed_drugs and observed_drugs[0] != "Untreated":
                issues.append(f"{cell_line_id}: Untreated is present but not first row.")

    return issues


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        return 2

    payload = load_collection(input_path)
    cell_lines, total_drug_rows, n_genes = summarize_collection(payload)
    issues = validate_collection(
        payload,
        expected_cell_lines=args.expected_cell_lines,
        check_untreated_first=args.check_untreated_first,
    )

    print(f"Loaded: {input_path}")
    print(f"Collection format: {payload.get('format', 'unknown')}")
    print(f"Cell lines: {len(cell_lines)}")
    print(f"Total drug rows across all cell lines: {total_drug_rows}")
    print(f"Max genes per cell line AnnData: {n_genes}")
    if cell_lines:
        print(f"First 5 cell lines: {cell_lines[:5]}")

    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1 if args.strict else 0

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())