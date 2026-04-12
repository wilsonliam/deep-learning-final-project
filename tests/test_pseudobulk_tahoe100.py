"""Tests for scripts/pseudobulk_tahoe100.py.

All tests use synthetic in-memory data — no HuggingFace network calls.
Run with:  pytest tests/test_pseudobulk_tahoe100.py -v
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

# Make the scripts directory importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pseudobulk_tahoe100 as pb
from pseudobulk_tahoe100 import (
    BlockAccumulator,
    GROUPBY_OBS_COLUMNS,
    OBS_INDEX_NAME,
    PseudobulkGroupKey,
    _densify_record,
    _log_first_record_latency,
    _iter_ordered_densified_records,
    build_target_cell_line_ids,
    main,
    run_pseudobulk,
    write_cell_line_output,
    write_outputs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_GENES_SMALL = 10  # tiny gene count for fast tests


def _make_cells(n_cells: int, n_genes: int = N_GENES_SMALL, seed: int = 42):
    """Return a list of dense expression vectors with known values."""
    rng = np.random.RandomState(seed)
    return [rng.rand(n_genes).astype(np.float32) for _ in range(n_cells)]


def _fake_stream(
    pairs: dict[PseudobulkGroupKey, list[np.ndarray]],
):
    """Yield (group_key, dense) from a dict of prepared cells."""
    for group_key, cells in pairs.items():
        for cell in cells:
            yield group_key, cell


def _group_key(
    cell_line_id: str,
    drug_key: str,
    *,
    drug: str | None = None,
    sample: str = "",
    plate: str = "",
    barcode_sub_lib_id: str = "",
) -> PseudobulkGroupKey:
    """Build a replicate-aware pseudobulk key for synthetic tests."""
    return PseudobulkGroupKey(
        cell_line_id=cell_line_id,
        drug_key=drug_key,
        drug=drug or drug_key,
        sample=sample,
        plate=plate,
        barcode_sub_lib_id=barcode_sub_lib_id,
    )


def _small_token_to_col(n_genes: int = N_GENES_SMALL) -> np.ndarray:
    """Identity mapping: token_id i → column i."""
    return np.arange(n_genes, dtype=np.int64)


def _raw_record(
    cell_line_id: str,
    drug: str,
    expr: np.ndarray,
    *,
    sample: str = "S0",
    plate: str = "P0",
    barcode_sub_lib_id: str = "",
) -> dict:
    """Encode a dense vector as a synthetic Tahoe-style raw record."""
    return {
        "cell_line_id": cell_line_id,
        "drug": drug,
        "sample": sample,
        "plate": plate,
        "BARCODE_SUB_LIB_ID": barcode_sub_lib_id,
        "genes": list(range(len(expr))),
        "expressions": expr.astype(np.float32).tolist(),
    }


def _sample_metadata_row(
    sample: str,
    drug: str,
    *,
    plate: str,
    drug_key: str | None = None,
) -> dict:
    """Build one synthetic Tahoe sample_metadata row."""
    return {
        "sample": sample,
        "drug": drug,
        "plate": plate,
        "drugname_drugconc": drug_key or drug,
    }


def _sample_metadata_lookup_from_raw_records(
    raw_records: list[dict],
) -> dict[str, pb.SampleMetadataEntry]:
    """Build a minimal sample_metadata lookup that matches synthetic raw rows."""
    rows_by_sample: dict[str, dict] = {}
    for record in raw_records:
        sample = str(record["sample"])
        row = _sample_metadata_row(
            sample,
            str(record["drug"]),
            plate=str(record["plate"]),
        )
        existing = rows_by_sample.get(sample)
        if existing is not None and existing != row:
            raise ValueError(
                f"Synthetic raw records reuse sample {sample!r} across multiple "
                "drug or plate values."
            )
        rows_by_sample[sample] = row
    return pb.build_sample_metadata_lookup(rows_by_sample.values())


def _run_ordered_pseudobulk(
    raw_records: list[dict],
    *,
    n_genes: int,
    block_size: int,
    num_workers: int,
    sample_metadata_by_sample: dict[str, pb.SampleMetadataEntry] | None = None,
    sample_size: int | None = None,
    cell_lines_whitelist: set[str] | None = None,
):
    """Run the ordered densification helper, then accumulate pseudobulk."""
    if sample_metadata_by_sample is None:
        sample_metadata_by_sample = _sample_metadata_lookup_from_raw_records(raw_records)
    stream = _iter_ordered_densified_records(
        raw_records,
        n_genes=n_genes,
        token_to_col=_small_token_to_col(n_genes),
        sample_metadata_by_sample=sample_metadata_by_sample,
        sample_size=sample_size,
        cell_lines_whitelist=cell_lines_whitelist,
        num_workers=num_workers,
    )
    return run_pseudobulk(
        stream,
        n_genes=n_genes,
        block_size=block_size,
        progress_every=0,
    )


class _RecordingProgressReporter:
    """Test reporter that records every progress callback."""

    def __init__(self):
        self.updates: list[tuple[int, int]] = []
        self.closed: tuple[int, int] | None = None

    def update(self, processed_cells: int, n_groups: int) -> None:
        self.updates.append((processed_cells, n_groups))

    def close(self, processed_cells: int, n_groups: int) -> None:
        self.closed = (processed_cells, n_groups)


class _FakeTqdmBar:
    """Tiny tqdm stand-in for progress-bar tests."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.updated: list[int] = []
        self.postfixes: list[tuple[str, bool]] = []
        self.refresh_count = 0
        self.closed = False

    def update(self, amount: int) -> None:
        self.updated.append(amount)

    def set_postfix_str(self, text: str, refresh: bool = False) -> None:
        self.postfixes.append((text, refresh))

    def refresh(self) -> None:
        self.refresh_count += 1

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# 1. test_accumulator_exact_boundary
# ---------------------------------------------------------------------------

def test_accumulator_exact_boundary():
    """400 cells, block_size=200 → 2 blocks, 0 dropped."""
    n_genes = N_GENES_SMALL
    block_size = 200
    cells = _make_cells(400, n_genes)
    acc = BlockAccumulator(n_genes=n_genes, block_size=block_size)
    for c in cells:
        acc.add_cell(c)

    assert acc.n_complete_blocks == 2
    assert acc.n_cells_seen == 400
    assert acc.n_cells_dropped == 0

    expected_block1 = sum(c for c in cells[:200])
    expected_block2 = sum(c for c in cells[200:400])
    expected = ((expected_block1 + expected_block2) / 2).astype(np.float32)
    result = acc.finalize()
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. test_accumulator_user_spec_1052
# ---------------------------------------------------------------------------

def test_accumulator_user_spec_1052():
    """User's worked example: 1052 cells → 5 blocks, 52 dropped."""
    n_genes = N_GENES_SMALL
    block_size = 200
    cells = _make_cells(1052, n_genes, seed=99)
    acc = BlockAccumulator(n_genes=n_genes, block_size=block_size)
    for c in cells:
        acc.add_cell(c)

    assert acc.n_complete_blocks == 5
    assert acc.n_cells_dropped == 52
    assert acc.n_cells_used == 1000

    # Ground truth: sum of first 1000 cells / 5
    expected = (
        sum(c for c in cells[:1000]) / 5
    ).astype(np.float32)
    result = acc.finalize()
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. test_accumulator_insufficient_cells
# ---------------------------------------------------------------------------

def test_accumulator_insufficient_cells():
    """199 cells, block_size=200 → 0 complete blocks, finalize returns None."""
    n_genes = N_GENES_SMALL
    cells = _make_cells(199, n_genes)
    acc = BlockAccumulator(n_genes=n_genes, block_size=200)
    for c in cells:
        acc.add_cell(c)

    assert acc.n_complete_blocks == 0
    assert acc.finalize() is None


# ---------------------------------------------------------------------------
# 4. test_multi_drug_isolation
# ---------------------------------------------------------------------------

def test_multi_drug_isolation():
    """Two drugs in same cell line stay isolated."""
    n_genes = N_GENES_SMALL
    block_size = 2  # small for speed
    cells_a = _make_cells(4, n_genes, seed=1)
    cells_b = _make_cells(4, n_genes, seed=2)

    stream = _fake_stream({
        _group_key("CL1", "DrugA", sample="S1", plate="P1", barcode_sub_lib_id="B1"): cells_a,
        _group_key("CL1", "DrugB", sample="S1", plate="P1", barcode_sub_lib_id="B1"): cells_b,
    })
    accs = run_pseudobulk(stream, n_genes=n_genes, block_size=block_size, progress_every=0)

    result_a = accs[_group_key("CL1", "DrugA", sample="S1", plate="P1", barcode_sub_lib_id="B1")].finalize()
    result_b = accs[_group_key("CL1", "DrugB", sample="S1", plate="P1", barcode_sub_lib_id="B1")].finalize()

    expected_a = (
        (sum(c for c in cells_a[:2]) + sum(c for c in cells_a[2:4])) / 2
    ).astype(np.float32)
    expected_b = (
        (sum(c for c in cells_b[:2]) + sum(c for c in cells_b[2:4])) / 2
    ).astype(np.float32)

    np.testing.assert_allclose(result_a, expected_a, rtol=1e-5)
    np.testing.assert_allclose(result_b, expected_b, rtol=1e-5)
    # Verify they're not accidentally equal.
    assert not np.allclose(result_a, result_b)


# ---------------------------------------------------------------------------
# 5. test_same_drug_samples_stay_isolated
# ---------------------------------------------------------------------------

def test_same_drug_samples_stay_isolated():
    """Same drug in different samples must not be merged into one pseudobulk row."""
    n_genes = N_GENES_SMALL
    block_size = 2
    cells_s1 = _make_cells(4, n_genes, seed=11)
    cells_s2 = _make_cells(4, n_genes, seed=22)

    key_s1 = _group_key("CL1", "DrugA", sample="S1", plate="P1", barcode_sub_lib_id="B1")
    key_s2 = _group_key("CL1", "DrugA", sample="S2", plate="P2", barcode_sub_lib_id="B2")

    stream = _fake_stream({
        key_s1: cells_s1,
        key_s2: cells_s2,
    })
    accs = run_pseudobulk(stream, n_genes=n_genes, block_size=block_size, progress_every=0)

    assert set(accs) == {key_s1, key_s2}

    expected_s1 = (
        (sum(c for c in cells_s1[:2]) + sum(c for c in cells_s1[2:4])) / 2
    ).astype(np.float32)
    expected_s2 = (
        (sum(c for c in cells_s2[:2]) + sum(c for c in cells_s2[2:4])) / 2
    ).astype(np.float32)

    np.testing.assert_allclose(accs[key_s1].finalize(), expected_s1, rtol=1e-5)
    np.testing.assert_allclose(accs[key_s2].finalize(), expected_s2, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5b. sample_metadata enrichment helpers
# ---------------------------------------------------------------------------

def test_build_sample_metadata_lookup_accepts_unique_samples():
    """Unique sample keys should build a lookup with concentration labels."""
    lookup = pb.build_sample_metadata_lookup(
        [
            _sample_metadata_row("S1", "DrugA", plate="P1", drug_key="DrugA_1uM"),
            _sample_metadata_row("S2", "DrugA", plate="P2", drug_key="DrugA_10uM"),
        ]
    )

    assert lookup["S1"].drug == "DrugA"
    assert lookup["S1"].plate == "P1"
    assert lookup["S1"].drugname_drugconc == "DrugA_1uM"


def test_build_sample_metadata_lookup_strips_whitespace():
    """Sample metadata should normalize leading/trailing whitespace."""
    lookup = pb.build_sample_metadata_lookup(
        [
            {
                "sample": " S1 ",
                "drug": " DrugA ",
                "plate": " P1 ",
                "drugname_drugconc": " DrugA_1uM ",
            }
        ]
    )

    assert list(lookup) == ["S1"]
    assert lookup["S1"].drug == "DrugA"
    assert lookup["S1"].plate == "P1"
    assert lookup["S1"].drugname_drugconc == "DrugA_1uM"


def test_build_sample_metadata_lookup_rejects_duplicate_samples():
    """Duplicate sample keys must fail fast."""
    with pytest.raises(RuntimeError, match="duplicate sample keys"):
        pb.build_sample_metadata_lookup(
            [
                _sample_metadata_row("S1", "DrugA", plate="P1", drug_key="DrugA_1uM"),
                _sample_metadata_row("S1", "DrugA", plate="P1", drug_key="DrugA_10uM"),
            ]
        )


def test_build_sample_metadata_lookup_requires_drugname_drugconc():
    """Missing concentration-aware labels should fail during metadata load."""
    with pytest.raises(RuntimeError, match="empty drugname_drugconc values"):
        pb.build_sample_metadata_lookup(
            [
                {
                    "sample": "S1",
                    "drug": "DrugA",
                    "plate": "P1",
                    "drugname_drugconc": "",
                }
            ]
        )


def test_build_target_cell_line_ids_normalizes_and_sorts():
    """Cell-line metadata should normalize whitespace and return sorted unique ids."""
    assert build_target_cell_line_ids(
        [
            {"Cell_ID_Cellosaur": " CVCL_0002 "},
            {"Cell_ID_Cellosaur": "CVCL_0001"},
            {"Cell_ID_Cellosaur": "CVCL_0002"},
        ]
    ) == ["CVCL_0001", "CVCL_0002"]


def test_build_target_cell_line_ids_requires_non_empty_values():
    """Empty cell_line_id metadata should fail fast."""
    with pytest.raises(RuntimeError, match="cell_line_id values must be non-empty"):
        build_target_cell_line_ids([{"Cell_ID_Cellosaur": " "}])


def test_normalize_record_with_sample_metadata_enriches_row():
    """Current-schema stream rows should be enriched before densification."""
    lookup = pb.build_sample_metadata_lookup(
        [_sample_metadata_row("S1", "DrugA", plate="P1", drug_key="DrugA_1uM")]
    )
    record = _raw_record("CL1", "DrugA", np.array([1.0, 2.0], dtype=np.float32), sample="S1", plate="P1")

    normalized = pb._normalize_record_with_sample_metadata(record, lookup)

    assert normalized["drug"] == "DrugA"
    assert normalized["plate"] == "P1"
    assert normalized["drugname_drugconc"] == "DrugA_1uM"


def test_normalize_record_with_sample_metadata_ignores_trailing_whitespace():
    """Stream/sample metadata comparisons should tolerate surrounding whitespace."""
    lookup = pb.build_sample_metadata_lookup(
        [
            {
                "sample": "S1",
                "drug": "selinexor ",
                "plate": "P1 ",
                "drugname_drugconc": " selinexor_1uM ",
            }
        ]
    )
    record = _raw_record(
        "CL1",
        " selinexor",
        np.array([1.0], dtype=np.float32),
        sample=" S1 ",
        plate="P1",
    )

    normalized = pb._normalize_record_with_sample_metadata(record, lookup)

    assert normalized["sample"] == "S1"
    assert normalized["drug"] == "selinexor"
    assert normalized["plate"] == "P1"
    assert normalized["drugname_drugconc"] == "selinexor_1uM"


def test_normalize_record_with_sample_metadata_requires_known_sample():
    """Rows that reference unknown samples should fail fast."""
    lookup = pb.build_sample_metadata_lookup(
        [_sample_metadata_row("S1", "DrugA", plate="P1", drug_key="DrugA_1uM")]
    )
    record = _raw_record("CL1", "DrugA", np.array([1.0], dtype=np.float32), sample="S2", plate="P1")

    with pytest.raises(RuntimeError, match="no matching sample_metadata row"):
        pb._normalize_record_with_sample_metadata(record, lookup)


def test_normalize_record_with_sample_metadata_rejects_drug_or_plate_conflicts():
    """Stream rows must agree with sample_metadata on drug and plate."""
    lookup = pb.build_sample_metadata_lookup(
        [_sample_metadata_row("S1", "DrugA", plate="P1", drug_key="DrugA_1uM")]
    )

    bad_drug = _raw_record("CL1", "DrugB", np.array([1.0], dtype=np.float32), sample="S1", plate="P1")
    with pytest.raises(RuntimeError, match="disagrees with sample_metadata drug"):
        pb._normalize_record_with_sample_metadata(bad_drug, lookup)

    bad_plate = _raw_record("CL1", "DrugA", np.array([1.0], dtype=np.float32), sample="S1", plate="P2")
    with pytest.raises(RuntimeError, match="disagrees with sample_metadata plate"):
        pb._normalize_record_with_sample_metadata(bad_plate, lookup)


# ---------------------------------------------------------------------------
# 5c. progress helpers
# ---------------------------------------------------------------------------

def test_resolve_progress_total_cells_prefers_sample_size():
    """Remaining sample-size budget should define the active per-line total."""
    assert pb._resolve_progress_total_cells(
        remaining_sample_size=2_000,
        progress_total_cells=9_999,
        target_cell_line_count=3,
    ) == (2_000, False)


def test_resolve_progress_total_cells_uses_override_for_single_target_run():
    """Manual total override should apply when exactly one target cell line is active."""
    assert pb._resolve_progress_total_cells(
        remaining_sample_size=None,
        progress_total_cells=12_345,
        target_cell_line_count=1,
    ) == (12_345, False)


def test_resolve_progress_total_cells_ignores_override_for_multi_target_run():
    """Multi-target runs should stay open-ended without a remaining sample budget."""
    assert pb._resolve_progress_total_cells(
        remaining_sample_size=None,
        progress_total_cells=99_999,
        target_cell_line_count=2,
    ) == (None, False)


def test_build_progress_reporter_auto_uses_bar_on_tty(monkeypatch):
    """Auto mode should pick the tqdm-backed reporter on an interactive stderr."""
    bars: list[_FakeTqdmBar] = []

    def fake_tqdm(*args, **kwargs):
        bar = _FakeTqdmBar(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(pb, "_TQDM_FACTORY", fake_tqdm)
    monkeypatch.setattr(pb, "_stderr_supports_live_progress", lambda stream=None: True)

    reporter = pb._build_progress_reporter(
        progress="auto",
        progress_every=25,
        total_cells=1_000,
        total_is_estimated=False,
    )

    assert isinstance(reporter, pb.TqdmProgressReporter)
    reporter.update(25, 3)
    reporter.close(25, 3)
    assert bars[0].kwargs["total"] == 1_000
    assert bars[0].postfixes[-1][0] == "groups=3"
    assert bars[0].closed is True


def test_build_progress_reporter_auto_uses_logs_without_tty(monkeypatch):
    """Auto mode should fall back to log progress off a TTY."""
    monkeypatch.setattr(pb, "_stderr_supports_live_progress", lambda stream=None: False)

    reporter = pb._build_progress_reporter(
        progress="auto",
        progress_every=25,
        total_cells=1_000,
        total_is_estimated=False,
    )

    assert isinstance(reporter, pb.LogProgressReporter)


def test_build_progress_reporter_explicit_log_and_off(monkeypatch):
    """Explicit log/off modes should bypass auto-detection."""
    monkeypatch.setattr(pb, "_stderr_supports_live_progress", lambda stream=None: True)

    log_reporter = pb._build_progress_reporter(
        progress="log",
        progress_every=25,
        total_cells=1_000,
        total_is_estimated=False,
    )
    off_reporter = pb._build_progress_reporter(
        progress="off",
        progress_every=25,
        total_cells=1_000,
        total_is_estimated=False,
    )

    assert isinstance(log_reporter, pb.LogProgressReporter)
    assert isinstance(off_reporter, pb.NullProgressReporter)


def test_run_pseudobulk_updates_progress_reporter_without_changing_results():
    """Progress callbacks should track processed cells/groups without changing math."""
    n_genes = N_GENES_SMALL
    block_size = 2
    cells = _make_cells(4, n_genes, seed=5)
    key = _group_key("CL1", "DrugA", sample="S1", plate="P1")
    reporter = _RecordingProgressReporter()

    accs = run_pseudobulk(
        _fake_stream({key: cells}),
        n_genes=n_genes,
        block_size=block_size,
        progress_every=0,
        progress_reporter=reporter,
    )

    assert reporter.updates == [(1, 1), (2, 1), (3, 1), (4, 1)]
    assert reporter.closed == (4, 1)
    expected = ((sum(cells[:2]) + sum(cells[2:4])) / 2).astype(np.float32)
    np.testing.assert_allclose(accs[key].finalize(), expected, rtol=1e-5)


def test_log_progress_reporter_emits_eta_and_estimated_total(monkeypatch, caplog):
    """Log mode should emit richer progress lines with ETA and estimate labeling."""
    caplog.set_level(logging.INFO, logger=pb.log.name)
    times = iter([100.0, 102.0])
    monkeypatch.setattr(pb.time, "monotonic", lambda: next(times))
    reporter = pb.LogProgressReporter(
        total_cells=10,
        total_is_estimated=True,
        progress_every=2,
    )

    reporter.update(2, 1)

    assert "Processed 2/~10 cells" in caplog.text
    assert "ETA" in caplog.text
    assert "estimated total" in caplog.text


# ---------------------------------------------------------------------------
# 6. test_sentinel_first_expression_dropped
# ---------------------------------------------------------------------------

def test_sentinel_first_expression_dropped():
    """Negative first expression → first gene/expression pair is dropped."""
    n_genes = 5
    # token_to_col: identity mapping for tokens 0..4
    token_to_col = np.arange(5, dtype=np.int64)

    record = {
        "cell_line_id": "CL1",
        "drug": "Drug",
        "drugname_drugconc": "Drug_1uM",
        "genes": [99, 0, 2, 4],       # 99 is the sentinel gene (skipped)
        "expressions": [-1.0, 1.5, 2.5, 3.5],  # -1.0 triggers sentinel drop
    }

    result = _densify_record(record, n_genes, token_to_col)
    assert result is not None
    group_key, dense = result
    assert group_key == _group_key("CL1", "Drug_1uM", drug="Drug")
    # After sentinel drop: genes=[0,2,4], exprs=[1.5, 2.5, 3.5]
    expected = np.array([1.5, 0.0, 2.5, 0.0, 3.5], dtype=np.float32)
    np.testing.assert_array_equal(dense, expected)


# ---------------------------------------------------------------------------
# 6. test_unknown_token_id_raises
# ---------------------------------------------------------------------------

def test_unknown_token_id_raises():
    """A token_id not in the vocab raises ValueError."""
    n_genes = 5
    token_to_col = np.arange(5, dtype=np.int64)

    record_out_of_range = {
        "cell_line_id": "CL1",
        "drug": "Drug",
        "drugname_drugconc": "Drug_1uM",
        "genes": [0, 999],   # 999 is out of range
        "expressions": [1.0, 2.0],
    }
    with pytest.raises(ValueError, match="exceed vocab size"):
        _densify_record(record_out_of_range, n_genes, token_to_col)

    # Also test a token that's in range but mapped to -1.
    token_to_col_sparse = np.full(10, -1, dtype=np.int64)
    token_to_col_sparse[0] = 0
    token_to_col_sparse[2] = 1
    record_unmapped = {
        "cell_line_id": "CL1",
        "drug": "Drug",
        "drugname_drugconc": "Drug_1uM",
        "genes": [0, 5],   # 5 maps to -1
        "expressions": [1.0, 2.0],
    }
    with pytest.raises(ValueError, match="Unknown token IDs"):
        _densify_record(record_unmapped, 2, token_to_col_sparse)


# ---------------------------------------------------------------------------
# 6b. test_negative_token_id_raises
# ---------------------------------------------------------------------------

def test_negative_token_id_raises():
    """Negative token IDs raise ValueError instead of silently wrapping."""
    n_genes = 5
    token_to_col = np.arange(5, dtype=np.int64)

    record_neg1 = {
        "cell_line_id": "CL1",
        "drug": "Drug",
        "drugname_drugconc": "Drug_1uM",
        "genes": [-1, 0, 2],
        "expressions": [1.0, 2.0, 3.0],
    }
    with pytest.raises(ValueError, match="Negative token IDs"):
        _densify_record(record_neg1, n_genes, token_to_col)

    record_large_neg = {
        "cell_line_id": "CL1",
        "drug": "Drug",
        "drugname_drugconc": "Drug_1uM",
        "genes": [0, -999],
        "expressions": [1.0, 2.0],
    }
    with pytest.raises(ValueError, match="Negative token IDs"):
        _densify_record(record_large_neg, n_genes, token_to_col)


# ---------------------------------------------------------------------------
# 7. test_end_to_end_writes_h5ad
# ---------------------------------------------------------------------------

def test_end_to_end_writes_h5ad(tmp_path):
    """Full pipeline preserves replicate-aware metadata in per-cell-line .h5ad files."""
    n_genes = N_GENES_SMALL
    block_size = 200
    gene_ids = np.array([f"ENSG{i:05d}" for i in range(n_genes)])

    pairs = {}
    for cl_idx, cl in enumerate(["CL_A", "CL_B"]):
        for drug_idx in range(3):
            drug = f"Drug{drug_idx}_1uM"
            plain_drug = f"Drug{drug_idx}"
            for rep_idx in range(2):
                sample = f"S{rep_idx}"
                plate = f"P{rep_idx}"
                barcode = f"B{rep_idx}"
                cells = _make_cells(
                    200,
                    n_genes,
                    seed=1000 * cl_idx + 100 * drug_idx + rep_idx,
                )
                pairs[
                    _group_key(
                        cl,
                        drug,
                        drug=plain_drug,
                        sample=sample,
                        plate=plate,
                        barcode_sub_lib_id=barcode,
                    )
                ] = cells

    stream = _fake_stream(pairs)
    accs = run_pseudobulk(stream, n_genes=n_genes, block_size=block_size, progress_every=0)

    run_meta = {"block_size": block_size, "source_dataset": "test"}
    written = write_outputs(accs, gene_ids, tmp_path, run_meta)

    assert len(written) == 2

    for path in written:
        assert path.exists()
        adata = ad.read_h5ad(path)
        assert adata.X.shape == (6, n_genes)
        assert adata.X.dtype == np.float32
        assert np.all(np.isfinite(adata.X))
        # Not all-zero for any replicate-aware row.
        assert np.all(np.abs(adata.X).sum(axis=1) > 0)
        assert adata.obs.index.name == OBS_INDEX_NAME
        assert "drug" in adata.obs.columns
        assert "drugname_drugconc" in adata.obs.columns
        assert "sample" in adata.obs.columns
        assert "plate" in adata.obs.columns
        assert "BARCODE_SUB_LIB_ID" in adata.obs.columns
        assert "replicate_id" in adata.obs.columns
        assert "n_cells_total" in adata.obs.columns
        assert "n_complete_blocks" in adata.obs.columns
        assert "n_cells_used" in adata.obs.columns
        assert "n_cells_dropped" in adata.obs.columns
        assert adata.var.index.name == "ensembl_id"
        assert adata.uns["block_size"] == block_size
        assert "cell_line_id" in adata.uns
        assert list(adata.uns["pseudobulk_groupby_columns"]) == list(GROUPBY_OBS_COLUMNS)
        assert (adata.obs["drug"].value_counts() == 2).all()
        # Each drug should now appear once per replicate-aware group.
        assert (adata.obs["drugname_drugconc"].value_counts() == 2).all()
        # All groups had exactly 200 cells → 1 block, 0 dropped.
        assert (adata.obs["n_complete_blocks"] == 1).all()
        assert (adata.obs["n_cells_dropped"] == 0).all()


def test_write_cell_line_output_returns_none_for_sparse_groups(tmp_path, caplog):
    """Single-cell-line writer should skip empty/sparse outputs without raising."""
    caplog.set_level(logging.WARNING, logger=pb.log.name)
    acc = BlockAccumulator(n_genes=N_GENES_SMALL, block_size=2)
    acc.add_cell(np.ones(N_GENES_SMALL, dtype=np.float32))
    key = _group_key("CL_A", "DrugA_1uM", drug="DrugA", sample="S1", plate="P1")

    path = write_cell_line_output(
        "CL_A",
        {key: acc},
        np.array([f"ENSG{i:05d}" for i in range(N_GENES_SMALL)]),
        tmp_path,
        {"block_size": 2, "source_dataset": "test"},
    )

    assert path is None
    assert "Cell line CL_A skipped" in caplog.text


def test_write_cell_line_output_writes_atomically(tmp_path):
    """Per-cell-line writes should land on the final path without stray temp files."""
    gene_ids = np.array([f"ENSG{i:05d}" for i in range(N_GENES_SMALL)])
    acc = BlockAccumulator(n_genes=N_GENES_SMALL, block_size=2)
    for _ in range(2):
        acc.add_cell(np.ones(N_GENES_SMALL, dtype=np.float32))
    key = _group_key("CL_A", "DrugA_1uM", drug="DrugA", sample="S1", plate="P1")

    path = write_cell_line_output(
        "CL_A",
        {key: acc},
        gene_ids,
        tmp_path,
        {"block_size": 2, "source_dataset": "test"},
    )

    assert path == tmp_path / "CL_A.h5ad"
    assert path.exists()
    assert list(tmp_path.glob(".CL_A.*.h5ad")) == []
    adata = ad.read_h5ad(path)
    assert adata.uns["cell_line_id"] == "CL_A"
    assert list(adata.obs["drugname_drugconc"]) == ["DrugA_1uM"]


# ---------------------------------------------------------------------------
# 8. test_parallel_matches_serial
# ---------------------------------------------------------------------------

def test_parallel_matches_serial():
    """Parallel (num_workers=2) produces identical results to serial (num_workers=0).

    Uses synthetic raw records to exercise the order-preserving densification
    helper without any HuggingFace network calls.
    """
    n_genes = N_GENES_SMALL
    block_size = 5

    # Build a reproducible synthetic dataset.
    rng = np.random.RandomState(7)
    raw_records: list[dict] = []
    for cl in ["CL_X", "CL_Y"]:
        for drug in ["D1_1uM", "D2_2uM"]:
            for rep_idx, (plate, barcode) in enumerate([("P1", "B1"), ("P2", "B2")], start=1):
                sample = f"{cl}_{drug}_S{rep_idx}"
                for _ in range(11):  # 11 cells → 2 blocks of 5, 1 dropped
                    raw_records.append(
                        _raw_record(
                            cl,
                            drug,
                            rng.rand(n_genes).astype(np.float32),
                            sample=sample,
                            plate=plate,
                            barcode_sub_lib_id=barcode,
                        )
                    )

    accs_serial = _run_ordered_pseudobulk(
        raw_records,
        n_genes=n_genes,
        block_size=block_size,
        num_workers=0,
    )
    accs_parallel = _run_ordered_pseudobulk(
        raw_records,
        n_genes=n_genes,
        block_size=block_size,
        num_workers=2,
    )

    assert set(accs_serial.keys()) == set(accs_parallel.keys())

    for key in accs_serial:
        acc_s = accs_serial[key]
        acc_p = accs_parallel[key]
        assert acc_s.n_cells_seen == acc_p.n_cells_seen
        assert acc_s.n_complete_blocks == acc_p.n_complete_blocks
        assert acc_s.n_cells_used == acc_p.n_cells_used
        assert acc_s.n_cells_dropped == acc_p.n_cells_dropped
        vec_s = acc_s.finalize()
        vec_p = acc_p.finalize()
        if vec_s is None:
            assert vec_p is None
        else:
            assert vec_p is not None
            np.testing.assert_array_equal(vec_s, vec_p)


def test_parallel_tail_drop_matches_serial_first_complete_block():
    """Parallel runs must drop the same trailing cells as the serial order."""
    n_genes = 1
    block_size = 5
    raw_records = [
        _raw_record("CL1", "DrugA", np.array([2**i], dtype=np.float32))
        for i in range(9)
    ]

    accs_serial = _run_ordered_pseudobulk(
        raw_records,
        n_genes=n_genes,
        block_size=block_size,
        num_workers=0,
    )
    accs_parallel = _run_ordered_pseudobulk(
        raw_records,
        n_genes=n_genes,
        block_size=block_size,
        num_workers=2,
    )

    key = _group_key("CL1", "DrugA", sample="S0", plate="P0")
    expected = np.array([31.0], dtype=np.float32)
    np.testing.assert_array_equal(accs_serial[key].finalize(), expected)
    np.testing.assert_array_equal(accs_parallel[key].finalize(), expected)
    assert accs_serial[key].n_cells_dropped == 4
    assert accs_parallel[key].n_cells_dropped == 4


def test_parallel_sample_size_matches_serial_cutoff_inside_group():
    """Sample-size truncation must keep the same first N valid cells."""
    n_genes = 1
    block_size = 5
    raw_records = [
        _raw_record("CL1", "DrugA", np.array([2**i], dtype=np.float32))
        for i in range(9)
    ]

    accs_serial = _run_ordered_pseudobulk(
        raw_records,
        n_genes=n_genes,
        block_size=block_size,
        num_workers=0,
        sample_size=6,
    )
    accs_parallel = _run_ordered_pseudobulk(
        raw_records,
        n_genes=n_genes,
        block_size=block_size,
        num_workers=2,
        sample_size=6,
    )

    key = _group_key("CL1", "DrugA", sample="S0", plate="P0")
    expected = np.array([31.0], dtype=np.float32)
    np.testing.assert_array_equal(accs_serial[key].finalize(), expected)
    np.testing.assert_array_equal(accs_parallel[key].finalize(), expected)
    assert accs_serial[key].n_cells_seen == 6
    assert accs_parallel[key].n_cells_seen == 6
    assert accs_serial[key].n_cells_used == 5
    assert accs_parallel[key].n_cells_used == 5
    assert accs_serial[key].n_cells_dropped == 1
    assert accs_parallel[key].n_cells_dropped == 1


# ---------------------------------------------------------------------------
# Edge-case: missing fields skip gracefully
# ---------------------------------------------------------------------------

def test_missing_fields_skip():
    """Records with empty cell_line_id or drugname_drugconc return None."""
    n_genes = 5
    token_to_col = np.arange(5, dtype=np.int64)

    record_missing_cl = {
        "cell_line_id": "",
        "drug": "Drug",
        "drugname_drugconc": "Drug_1uM",
        "genes": [0],
        "expressions": [1.0],
    }
    assert _densify_record(record_missing_cl, n_genes, token_to_col) is None

    record_missing_drug = {
        "cell_line_id": "CL1",
        "drug": "Drug",
        "drugname_drugconc": "",
        "genes": [0],
        "expressions": [1.0],
    }
    assert _densify_record(record_missing_drug, n_genes, token_to_col) is None


# ---------------------------------------------------------------------------
# Edge-case: duplicate gene tokens raise
# ---------------------------------------------------------------------------

def test_duplicate_gene_tokens_raises():
    """Duplicate gene tokens in a record raise ValueError."""
    n_genes = 5
    token_to_col = np.arange(5, dtype=np.int64)

    record = {
        "cell_line_id": "CL1",
        "drug": "Drug",
        "drugname_drugconc": "Drug_1uM",
        "genes": [0, 2, 0],          # token 0 appears twice
        "expressions": [1.0, 2.0, 9.0],
    }
    with pytest.raises(ValueError, match="Duplicate gene tokens"):
        _densify_record(record, n_genes, token_to_col)


class _FakeStreamDataset:
    """Tiny iterable that exposes a ``features`` attribute like HF datasets."""

    def __init__(self, rows=None, *, features=None):
        self._rows = list(rows or [])
        self.features = features

    def __iter__(self):
        return iter(self._rows)


def _expression_stream_features(*, missing_columns: tuple[str, ...] = ()):
    """Build a minimal Tahoe expression stream schema for offline tests."""
    missing = set(missing_columns)
    columns = [
        column
        for column in pb.REQUIRED_EXPRESSION_STREAM_COLUMNS
        if column not in missing
    ]
    return {column: column for column in columns}


def _install_main_fakes(
    monkeypatch,
    *,
    stream_features=None,
    stream_rows=None,
    sample_metadata_rows=None,
    cell_line_metadata_rows=None,
    run_pseudobulk_side_effect=None,
    write_cell_line_output_side_effect=None,
):
    """Patch networked entrypoints so ``main`` stays fully offline in tests."""
    load_calls = []
    write_calls = []
    if stream_features is None:
        stream_features = _expression_stream_features()
    stream_rows = list(stream_rows or [])
    sample_metadata_rows = list(
        sample_metadata_rows
        or [_sample_metadata_row("S0", "Drug0", plate="P0", drug_key="Drug0_1uM")]
    )
    cell_line_metadata_rows = list(
        cell_line_metadata_rows
        or [{"Cell_ID_Cellosaur": " CL_B "}, {"Cell_ID_Cellosaur": "CL_A"}]
    )

    def fake_load_dataset(path, *args, **kwargs):
        load_calls.append((path, kwargs.copy()))
        if kwargs.get("name") == "gene_metadata":
            return [
                {"token_id": 0, "ensembl_id": "ENSG00000"},
                {"token_id": 1, "ensembl_id": "ENSG00001"},
            ]
        if kwargs.get("name") == "sample_metadata":
            return sample_metadata_rows
        if kwargs.get("name") == "cell_line_metadata":
            return cell_line_metadata_rows
        return _FakeStreamDataset(stream_rows, features=stream_features)

    def fake_run_pseudobulk(*args, **kwargs):
        if run_pseudobulk_side_effect is not None:
            return run_pseudobulk_side_effect(*args, **kwargs)

        filters = load_calls[-1][1].get("filters") or [("cell_line_id", "in", ["CL_A"])]
        cell_line_id = filters[0][2][0]
        dense = np.ones(kwargs["n_genes"], dtype=np.float32)
        acc = BlockAccumulator(
            n_genes=kwargs["n_genes"],
            block_size=kwargs["block_size"],
        )
        for _ in range(kwargs["block_size"]):
            acc.add_cell(dense)
        key = _group_key(
            cell_line_id,
            f"{cell_line_id}_1uM",
            drug=f"{cell_line_id}_drug",
            sample=f"{cell_line_id}_sample",
            plate="P0",
            barcode_sub_lib_id="B0",
        )
        return {key: acc}

    def fake_write_cell_line_output(cell_line_id, accumulators, gene_ids, output_dir, run_meta):
        write_calls.append((cell_line_id, output_dir, run_meta.copy()))
        if write_cell_line_output_side_effect is not None:
            return write_cell_line_output_side_effect(
                cell_line_id,
                accumulators,
                gene_ids,
                output_dir,
                run_meta,
            )
        return output_dir / f"{cell_line_id}.h5ad"

    monkeypatch.setattr(pb, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(pb, "run_pseudobulk", fake_run_pseudobulk)
    monkeypatch.setattr(pb, "write_cell_line_output", fake_write_cell_line_output)
    return load_calls, write_calls


def test_main_forwards_hf_token_to_all_dataset_loads(monkeypatch, caplog):
    """Explicit HF_TOKEN should be forwarded to every Tahoe dataset load."""
    load_calls, _ = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    caplog.set_level(logging.INFO, logger=pb.log.name)

    main(["--smoke", "--num-workers", "0"])

    assert len(load_calls) == 6
    assert all(call[0] == pb.DATASET_NAME for call in load_calls)
    assert all(call[1]["token"] == "hf_secret_value" for call in load_calls)
    assert "Using Hugging Face env token from HF_TOKEN." in caplog.text
    assert "hf_secret_value" not in caplog.text


def test_main_accepts_legacy_hf_token_env_with_warning(monkeypatch, caplog):
    """Legacy env var still works, but it should warn and prefer the new name."""
    load_calls, _ = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy_secret")
    caplog.set_level(logging.INFO, logger=pb.log.name)

    main(["--smoke", "--num-workers", "0"])

    assert len(load_calls) == 6
    assert all(call[1]["token"] == "legacy_secret" for call in load_calls)
    assert "deprecated; prefer HF_TOKEN" in caplog.text
    assert "Using Hugging Face env token from HUGGING_FACE_HUB_TOKEN." in caplog.text
    assert "legacy_secret" not in caplog.text


def test_main_without_env_token_uses_default_auth_resolution(monkeypatch, caplog):
    """No env token should keep default Hugging Face auth behavior untouched."""
    load_calls, _ = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    caplog.set_level(logging.INFO, logger=pb.log.name)

    main(["--smoke", "--num-workers", "0"])

    assert len(load_calls) == 6
    assert all("token" not in call[1] for call in load_calls)
    assert (
        "No Hugging Face env token configured; relying on saved login or anonymous access."
        in caplog.text
    )


def test_main_forwards_hf_cache_dir_to_all_dataset_loads(monkeypatch, tmp_path):
    """CLI cache-dir override should be shared across every Tahoe dataset load."""
    load_calls, _ = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    cache_dir = tmp_path / "hf_cache"

    main(["--smoke", "--num-workers", "0", "--hf-cache-dir", str(cache_dir)])

    assert len(load_calls) == 6
    assert all(call[1]["cache_dir"] == str(cache_dir) for call in load_calls)


def test_main_accepts_current_expression_schema_and_requests_columns(monkeypatch):
    """Current Tahoe expression schema should preflight, then open a pruned stream."""
    load_calls, write_calls = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    main(["--smoke", "--num-workers", "0", "--cell-lines", "CL_A"])

    assert len(load_calls) == 4
    assert load_calls[0][1]["name"] == "gene_metadata"
    assert load_calls[1][1]["name"] == "sample_metadata"
    preflight_kwargs = load_calls[2][1]
    stream_kwargs = load_calls[3][1]
    assert "columns" not in preflight_kwargs
    assert "filters" not in preflight_kwargs
    assert stream_kwargs["columns"] == list(pb.REQUIRED_EXPRESSION_STREAM_COLUMNS)
    assert stream_kwargs["filters"] == [("cell_line_id", "in", ["CL_A"])]
    assert [call[0] for call in write_calls] == ["CL_A"]


def test_main_adds_cell_line_filter_to_expression_stream_load(monkeypatch):
    """A cell-line whitelist should be forwarded as a Parquet filter."""
    load_calls, write_calls = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    main(["--smoke", "--num-workers", "0", "--cell-lines", "CL_B", "CL_A"])

    assert len(load_calls) == 5
    assert load_calls[2][1].get("streaming") is True
    stream_kwargs_a = load_calls[3][1]
    stream_kwargs_b = load_calls[4][1]
    assert stream_kwargs_a["filters"] == [("cell_line_id", "in", ["CL_A"])]
    assert stream_kwargs_b["filters"] == [("cell_line_id", "in", ["CL_B"])]
    assert [call[0] for call in write_calls] == ["CL_A", "CL_B"]


def test_main_without_whitelist_loads_cell_line_metadata_and_processes_each_line(monkeypatch):
    """Unfiltered runs should resolve target ids from cell_line_metadata and stream one line at a time."""
    load_calls, write_calls = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    main(["--smoke", "--num-workers", "0"])

    assert len(load_calls) == 6
    assert load_calls[2][1]["name"] == "cell_line_metadata"
    assert load_calls[4][1]["filters"] == [("cell_line_id", "in", ["CL_A"])]
    assert load_calls[5][1]["filters"] == [("cell_line_id", "in", ["CL_B"])]
    assert [call[0] for call in write_calls] == ["CL_A", "CL_B"]


def test_main_fails_fast_when_expression_stream_misses_current_required_column(monkeypatch):
    """Schema preflight should still reject genuinely incompatible expression rows."""
    load_calls, _ = _install_main_fakes(
        monkeypatch,
        stream_features=_expression_stream_features(missing_columns=("sample",)),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="missing required columns \\['sample'\\]"):
        main(["--smoke", "--num-workers", "0"])

    assert len(load_calls) == 4


def test_main_fails_fast_when_sample_metadata_is_incomplete(monkeypatch):
    """Startup should abort before streaming when sample_metadata is incomplete."""
    load_calls, _ = _install_main_fakes(
        monkeypatch,
        sample_metadata_rows=[
            {
                "sample": "S0",
                "drug": "Drug0",
                "plate": "P0",
                "drugname_drugconc": "",
            }
        ],
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="empty drugname_drugconc values"):
        main(["--smoke", "--num-workers", "0"])

    assert len(load_calls) == 2


def test_main_logs_xet_and_timeout_envs(monkeypatch, caplog):
    """HF/Xet-related runtime env vars should be visible in startup logs."""
    _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "1")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "45")
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    monkeypatch.setenv("HF_DEBUG", "1")
    caplog.set_level(logging.INFO, logger=pb.log.name)

    main(["--smoke", "--num-workers", "0"])

    assert "HF_XET_HIGH_PERFORMANCE=1" in caplog.text
    assert "HF_HUB_ETAG_TIMEOUT=45" in caplog.text
    assert "HF_HUB_DOWNLOAD_TIMEOUT=120" in caplog.text
    assert "HF_DEBUG=1" in caplog.text
    assert "HF_HUB_ENABLE_HF_TRANSFER is deprecated" in caplog.text


def test_main_logs_progress_phases(monkeypatch, caplog):
    """Startup logs should call out the four high-level pipeline phases."""
    _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    caplog.set_level(logging.INFO, logger=pb.log.name)

    main(["--smoke", "--num-workers", "0"])

    assert "Phase 1/4: loading gene metadata" in caplog.text
    assert "Phase 2/4: loading sample metadata" in caplog.text
    assert "Phase 3/4: schema preflight" in caplog.text
    assert "Phase 4/4: streaming accumulation" in caplog.text


def test_main_logs_outer_cell_line_progress(monkeypatch, caplog):
    """Per-cell-line runs should log deterministic outer progress counters."""
    _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    caplog.set_level(logging.INFO, logger=pb.log.name)

    main(["--smoke", "--num-workers", "0"])

    assert "Cell line 1/2: CL_A (processing)" in caplog.text
    assert "Cell line 2/2: CL_B (processing)" in caplog.text


def test_main_skips_existing_outputs_without_restreaming(monkeypatch, tmp_path):
    """Existing final outputs should be skipped by default on rerun."""
    load_calls, write_calls = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "CL_A.h5ad").write_text("existing")

    main(
        [
            "--num-workers",
            "0",
            "--cell-lines",
            "CL_A",
            "CL_B",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert len(load_calls) == 4
    assert load_calls[3][1]["filters"] == [("cell_line_id", "in", ["CL_B"])]
    assert [call[0] for call in write_calls] == ["CL_B"]


def test_main_overwrite_existing_recomputes(monkeypatch, tmp_path):
    """--overwrite-existing should force processing even when outputs already exist."""
    load_calls, write_calls = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "CL_A.h5ad").write_text("existing")

    main(
        [
            "--num-workers",
            "0",
            "--cell-lines",
            "CL_A",
            "CL_B",
            "--output-dir",
            str(output_dir),
            "--overwrite-existing",
        ]
    )

    assert len(load_calls) == 5
    assert load_calls[3][1]["filters"] == [("cell_line_id", "in", ["CL_A"])]
    assert load_calls[4][1]["filters"] == [("cell_line_id", "in", ["CL_B"])]
    assert [call[0] for call in write_calls] == ["CL_A", "CL_B"]


def test_main_stops_later_cell_lines_when_global_sample_size_is_exhausted(monkeypatch, tmp_path):
    """The global sample-size budget should stop scheduling later cell lines."""
    load_calls, write_calls = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    main(
        [
            "--num-workers",
            "0",
            "--cell-lines",
            "CL_A",
            "CL_B",
            "--sample-size",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert len(load_calls) == 4
    assert load_calls[3][1]["filters"] == [("cell_line_id", "in", ["CL_A"])]
    assert [call[0] for call in write_calls] == ["CL_A"]


def test_main_warns_when_progress_total_is_ignored_for_multi_target_runs(monkeypatch, caplog):
    """Global progress-total overrides should warn when multiple cell lines are scheduled."""
    _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    caplog.set_level(logging.INFO, logger=pb.log.name)

    main(
        [
            "--smoke",
            "--num-workers",
            "0",
            "--cell-lines",
            "CL_A",
            "CL_B",
            "--progress-total-cells",
            "123",
        ]
    )

    assert "--progress-total-cells is ignored for multi-cell-line runs" in caplog.text


def test_first_record_latency_logs(monkeypatch, caplog):
    """The stream wrapper should log when the first record arrives."""
    caplog.set_level(logging.INFO, logger=pb.log.name)
    times = iter([12.5])
    monkeypatch.setattr(pb.time, "monotonic", lambda: next(times))

    records = list(
        _log_first_record_latency(
            [{"row": 1}, {"row": 2}],
            started_at=10.0,
            stream_name="test stream",
        )
    )

    assert records == [{"row": 1}, {"row": 2}]
    assert "First record received from test stream after 2.5s" in caplog.text
