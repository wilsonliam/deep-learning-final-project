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
    main,
    run_pseudobulk,
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
    sample: str = "",
    plate: str = "",
    barcode_sub_lib_id: str = "",
) -> PseudobulkGroupKey:
    """Build a replicate-aware pseudobulk key for synthetic tests."""
    return PseudobulkGroupKey(
        cell_line_id=cell_line_id,
        drug_key=drug_key,
        sample=sample,
        plate=plate,
        barcode_sub_lib_id=barcode_sub_lib_id,
    )


def _small_token_to_col(n_genes: int = N_GENES_SMALL) -> np.ndarray:
    """Identity mapping: token_id i → column i."""
    return np.arange(n_genes, dtype=np.int64)


def _raw_record(
    cell_line_id: str,
    drug_key: str,
    expr: np.ndarray,
    *,
    sample: str = "",
    plate: str = "",
    barcode_sub_lib_id: str = "",
) -> dict:
    """Encode a dense vector as a synthetic Tahoe-style raw record."""
    return {
        "cell_line_id": cell_line_id,
        "drugname_drugconc": drug_key,
        "sample": sample,
        "plate": plate,
        "BARCODE_SUB_LIB_ID": barcode_sub_lib_id,
        "genes": list(range(len(expr))),
        "expressions": expr.astype(np.float32).tolist(),
    }


def _run_ordered_pseudobulk(
    raw_records: list[dict],
    *,
    n_genes: int,
    block_size: int,
    num_workers: int,
    sample_size: int | None = None,
    cell_lines_whitelist: set[str] | None = None,
):
    """Run the ordered densification helper, then accumulate pseudobulk."""
    stream = _iter_ordered_densified_records(
        raw_records,
        n_genes=n_genes,
        token_to_col=_small_token_to_col(n_genes),
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
# 6. test_sentinel_first_expression_dropped
# ---------------------------------------------------------------------------

def test_sentinel_first_expression_dropped():
    """Negative first expression → first gene/expression pair is dropped."""
    n_genes = 5
    # token_to_col: identity mapping for tokens 0..4
    token_to_col = np.arange(5, dtype=np.int64)

    record = {
        "cell_line_id": "CL1",
        "drugname_drugconc": "Drug_1uM",
        "genes": [99, 0, 2, 4],       # 99 is the sentinel gene (skipped)
        "expressions": [-1.0, 1.5, 2.5, 3.5],  # -1.0 triggers sentinel drop
    }

    result = _densify_record(record, n_genes, token_to_col)
    assert result is not None
    group_key, dense = result
    assert group_key == _group_key("CL1", "Drug_1uM")
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
        "drugname_drugconc": "Drug_1uM",
        "genes": [-1, 0, 2],
        "expressions": [1.0, 2.0, 3.0],
    }
    with pytest.raises(ValueError, match="Negative token IDs"):
        _densify_record(record_neg1, n_genes, token_to_col)

    record_large_neg = {
        "cell_line_id": "CL1",
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
        # Each drug should now appear once per replicate-aware group.
        assert (adata.obs["drugname_drugconc"].value_counts() == 2).all()
        # All groups had exactly 200 cells → 1 block, 0 dropped.
        assert (adata.obs["n_complete_blocks"] == 1).all()
        assert (adata.obs["n_cells_dropped"] == 0).all()


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
            for sample, plate, barcode in [("S1", "P1", "B1"), ("S2", "P2", "B2")]:
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

    key = _group_key("CL1", "DrugA")
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

    key = _group_key("CL1", "DrugA")
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
        "drugname_drugconc": "Drug_1uM",
        "genes": [0],
        "expressions": [1.0],
    }
    assert _densify_record(record_missing_cl, n_genes, token_to_col) is None

    record_missing_drug = {
        "cell_line_id": "CL1",
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
        "drugname_drugconc": "Drug_1uM",
        "genes": [0, 2, 0],          # token 0 appears twice
        "expressions": [1.0, 2.0, 9.0],
    }
    with pytest.raises(ValueError, match="Duplicate gene tokens"):
        _densify_record(record, n_genes, token_to_col)


def _install_main_fakes(monkeypatch):
    """Patch networked entrypoints so ``main`` stays fully offline in tests."""
    load_calls = []

    def fake_load_dataset(path, *args, **kwargs):
        load_calls.append((path, kwargs.copy()))
        if kwargs.get("name") == "gene_metadata":
            return [
                {"token_id": 0, "ensembl_id": "ENSG00000"},
                {"token_id": 1, "ensembl_id": "ENSG00001"},
            ]
        return iter(())

    monkeypatch.setattr(pb, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(pb, "run_pseudobulk", lambda *args, **kwargs: {})
    monkeypatch.setattr(pb, "write_outputs", lambda *args, **kwargs: [])
    return load_calls


def test_main_forwards_hf_token_to_both_dataset_loads(monkeypatch, caplog):
    """Explicit HF_TOKEN should be forwarded to both remote dataset loads."""
    load_calls = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    caplog.set_level(logging.INFO, logger=pb.log.name)

    main(["--smoke", "--num-workers", "0"])

    assert len(load_calls) == 2
    assert [call[0] for call in load_calls] == [pb.DATASET_NAME, pb.DATASET_NAME]
    assert all(call[1]["token"] == "hf_secret_value" for call in load_calls)
    assert "Using Hugging Face env token from HF_TOKEN." in caplog.text
    assert "hf_secret_value" not in caplog.text


def test_main_accepts_legacy_hf_token_env_with_warning(monkeypatch, caplog):
    """Legacy env var still works, but it should warn and prefer the new name."""
    load_calls = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy_secret")
    caplog.set_level(logging.INFO, logger=pb.log.name)

    main(["--smoke", "--num-workers", "0"])

    assert len(load_calls) == 2
    assert all(call[1]["token"] == "legacy_secret" for call in load_calls)
    assert "deprecated; prefer HF_TOKEN" in caplog.text
    assert "Using Hugging Face env token from HUGGING_FACE_HUB_TOKEN." in caplog.text
    assert "legacy_secret" not in caplog.text


def test_main_without_env_token_uses_default_auth_resolution(monkeypatch, caplog):
    """No env token should keep default Hugging Face auth behavior untouched."""
    load_calls = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    caplog.set_level(logging.INFO, logger=pb.log.name)

    main(["--smoke", "--num-workers", "0"])

    assert len(load_calls) == 2
    assert all("token" not in call[1] for call in load_calls)
    assert (
        "No Hugging Face env token configured; relying on saved login or anonymous access."
        in caplog.text
    )


def test_main_forwards_hf_cache_dir_to_both_dataset_loads(monkeypatch, tmp_path):
    """CLI cache-dir override should be shared across both dataset loads."""
    load_calls = _install_main_fakes(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    cache_dir = tmp_path / "hf_cache"

    main(["--smoke", "--num-workers", "0", "--hf-cache-dir", str(cache_dir)])

    assert len(load_calls) == 2
    assert all(call[1]["cache_dir"] == str(cache_dir) for call in load_calls)


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
