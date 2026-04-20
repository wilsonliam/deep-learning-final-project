import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.stats import mannwhitneyu
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

SPLIT_NAMES = ("train", "val", "test")
SUPPORTED_SPLIT_MODES = {"drug_blind", "tumor_blind", "mixed"}
SUPPORTED_TARGET_MODES = {"delta"}
SUPPORTED_PLOT_SAMPLE_STRATEGIES = {"seeded_random"}
SUPPORTED_EMBEDDING_METHODS = {"pca"}
MIN_STANDARD_DEVIATION = 1e-6


def get_project_root():
    for base_path in [Path.cwd(), *Path.cwd().parents]:
        if (base_path / "pyproject.toml").exists():
            return base_path
    return Path.cwd()


PROJECT_ROOT = get_project_root()


def build_project_path(relative_path):
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_target_gene_indices(input_gene_ids, target_gene_ids):
    input_gene_to_index = {str(gene_id): idx for idx, gene_id in enumerate(input_gene_ids)}
    target_gene_indices = []
    missing_target_gene_ids = []

    for gene_id in target_gene_ids:
        normalized_gene_id = str(gene_id)
        if normalized_gene_id not in input_gene_to_index:
            missing_target_gene_ids.append(normalized_gene_id)
        else:
            target_gene_indices.append(input_gene_to_index[normalized_gene_id])

    if missing_target_gene_ids:
        raise ValueError(
            f"Target genes were not found in the DMSO input gene space: {missing_target_gene_ids[:5]}"
        )

    return np.asarray(target_gene_indices, dtype=np.int64)


def resolve_project_path(relative_path):
    candidate_path = build_project_path(relative_path)
    if candidate_path.exists():
        return candidate_path
    raise FileNotFoundError(f"Could not find {relative_path} from {PROJECT_ROOT}")


def load_arrow_table(arrow_path):
    with pa.memory_map(str(arrow_path), "r") as source:
        try:
            return ipc.open_file(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return ipc.open_stream(source).read_all()


def load_cached_cell_line_metadata():
    arrow_candidates = sorted(
        Path.home().glob(
            ".cache/huggingface/datasets/vevotx___tahoe-100_m/cell_line_metadata/0.0.0/*/tahoe-100_m-train.arrow"
        )
    )
    if not arrow_candidates:
        raise FileNotFoundError(
            "Could not find cached Tahoe cell_line_metadata Arrow file in the Hugging Face cache."
        )

    arrow_path = arrow_candidates[-1]
    raw_df = load_arrow_table(arrow_path).to_pandas().loc[:, ["Cell_ID_Cellosaur", "cell_name", "Organ"]].copy()
    raw_df.columns = ["cell_line", "cell_name", "organ"]

    for column in raw_df.columns:
        raw_df[column] = raw_df[column].astype(str).str.strip()

    consistency_df = raw_df.groupby("cell_line", dropna=False).agg(
        cell_name_nunique=("cell_name", lambda s: s.nunique(dropna=False)),
        organ_nunique=("organ", lambda s: s.nunique(dropna=False)),
    )
    inconsistent_df = consistency_df.loc[
        (consistency_df["cell_name_nunique"] != 1)
        | (consistency_df["organ_nunique"] != 1)
    ]
    if not inconsistent_df.empty:
        raise ValueError(
            "Some Cellosaur IDs map to multiple cell_name/Organ values; cannot build a stable cell-line metadata table."
        )

    cell_line_metadata_df = (
        raw_df.groupby("cell_line", as_index=False, dropna=False)
        .first()
        .sort_values("cell_line", ignore_index=True)
    )
    return cell_line_metadata_df, arrow_path


def validate_split_config(split_mode, split_fractions):
    if split_mode not in SUPPORTED_SPLIT_MODES:
        raise ValueError(f"Unsupported SPLIT_MODE: {split_mode}")

    if set(split_fractions) != set(SPLIT_NAMES):
        raise ValueError(
            f"SPLIT_FRACTIONS must contain exactly {SPLIT_NAMES}; got {tuple(split_fractions)}"
        )

    if any(float(split_fractions[name]) <= 0 for name in SPLIT_NAMES):
        raise ValueError("All split fractions must be positive.")

    total_fraction = sum(float(split_fractions[name]) for name in SPLIT_NAMES)
    if not math.isclose(total_fraction, 1.0, abs_tol=1e-8):
        raise ValueError(f"Split fractions must sum to 1.0; got {total_fraction}")


def validate_preprocessing_config(target_mode, preprocess_batch_size):
    if target_mode not in SUPPORTED_TARGET_MODES:
        raise ValueError(f"Unsupported TARGET_MODE: {target_mode}")
    if int(preprocess_batch_size) <= 0:
        raise ValueError("PREPROCESS_BATCH_SIZE must be positive.")


def validate_training_config(
    target_mode,
    preprocess_batch_size,
    learning_rate,
    weight_decay,
    max_epochs,
    early_stopping_patience,
):
    validate_preprocessing_config(target_mode, preprocess_batch_size)
    if float(learning_rate) <= 0:
        raise ValueError("LEARNING_RATE must be positive.")
    if float(weight_decay) < 0:
        raise ValueError("WEIGHT_DECAY must be non-negative.")
    if int(max_epochs) <= 0:
        raise ValueError("MAX_EPOCHS must be positive.")
    if int(early_stopping_patience) < 0:
        raise ValueError("EARLY_STOPPING_PATIENCE must be non-negative.")


def validate_plot_config(plot_test_sample_count, plot_test_sample_strategy, embedding_method, n_embedding_components):
    if int(plot_test_sample_count) <= 0:
        raise ValueError("PLOT_TEST_SAMPLE_COUNT must be positive.")
    if plot_test_sample_strategy not in SUPPORTED_PLOT_SAMPLE_STRATEGIES:
        raise ValueError(f"Unsupported PLOT_TEST_SAMPLE_STRATEGY: {plot_test_sample_strategy}")
    if embedding_method not in SUPPORTED_EMBEDDING_METHODS:
        raise ValueError(f"Unsupported EMBEDDING_METHOD: {embedding_method}")
    if int(n_embedding_components) != 2:
        raise ValueError("N_EMBEDDING_COMPONENTS must be 2 for the paired plotting cell.")


def compute_target_row_counts(n_rows, split_fractions):
    val_rows = int(round(float(split_fractions["val"]) * n_rows))
    test_rows = int(round(float(split_fractions["test"]) * n_rows))
    train_rows = int(n_rows - val_rows - test_rows)

    if min(train_rows, val_rows, test_rows) <= 0:
        raise ValueError(
            f"Split fractions produced an empty split: train={train_rows}, val={val_rows}, test={test_rows}"
        )

    return {"train": train_rows, "val": val_rows, "test": test_rows}


def build_row_slices(n_rows, batch_size, min_last_batch_size=1):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    slices = []
    start = 0
    while start < n_rows:
        end = min(start + batch_size, n_rows)
        slices.append((start, end))
        start = end

    if len(slices) >= 2 and (slices[-1][1] - slices[-1][0]) < min_last_batch_size:
        previous_start, _ = slices[-2]
        _, final_end = slices[-1]
        slices[-2] = (previous_start, final_end)
        slices.pop()

    return slices


def assign_group_blind_splits(frame, group_col, split_fractions, seed):
    target_counts = compute_target_row_counts(len(frame), split_fractions)
    group_sizes = frame.groupby(group_col, dropna=False).size().reset_index(name="row_count")

    rng = np.random.default_rng(seed)
    group_sizes["shuffle_order"] = rng.permutation(len(group_sizes))
    group_sizes = group_sizes.sort_values(
        ["row_count", "shuffle_order"],
        ascending=[False, True],
        kind="stable",
        ignore_index=True,
    )

    assigned_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    group_to_split = {}

    for group_idx, row in group_sizes.iterrows():
        remaining_groups = len(group_sizes) - group_idx
        empty_splits = [split_name for split_name in SPLIT_NAMES if assigned_counts[split_name] == 0]
        if empty_splits and remaining_groups == len(empty_splits):
            candidate_splits = tuple(empty_splits)
        else:
            candidate_splits = SPLIT_NAMES

        group_size = int(row["row_count"])

        def assignment_score(split_name):
            projected = assigned_counts[split_name] + group_size
            target = target_counts[split_name]
            return (
                projected > target,
                abs(target - projected),
                assigned_counts[split_name] / max(target, 1),
                SPLIT_NAMES.index(split_name),
            )

        chosen_split = min(candidate_splits, key=assignment_score)
        group_to_split[row[group_col]] = chosen_split
        assigned_counts[chosen_split] += group_size

    split_series = frame[group_col].map(group_to_split)
    if split_series.isna().any():
        raise ValueError(f"Failed to assign every {group_col} to a split.")

    return split_series.astype(str)


def assign_mixed_split(frame, split_fractions, seed):
    target_counts = compute_target_row_counts(len(frame), split_fractions)
    split_assignments = np.full(len(frame), "train", dtype=object)
    remaining_drug_counts = frame["drug"].value_counts().to_dict()
    remaining_cell_line_counts = frame["cell_line"].value_counts().to_dict()
    drugs = frame["drug"].to_numpy()
    cell_lines = frame["cell_line"].to_numpy()

    rng = np.random.default_rng(seed)
    for split_name in ("val", "test"):
        target_rows = target_counts[split_name]
        assigned_rows = 0

        for row_idx in rng.permutation(len(frame)):
            if split_assignments[row_idx] != "train":
                continue

            drug = drugs[row_idx]
            cell_line = cell_lines[row_idx]
            if remaining_drug_counts[drug] <= 1 or remaining_cell_line_counts[cell_line] <= 1:
                continue

            split_assignments[row_idx] = split_name
            remaining_drug_counts[drug] -= 1
            remaining_cell_line_counts[cell_line] -= 1
            assigned_rows += 1

            if assigned_rows >= target_rows:
                break

        if assigned_rows < target_rows:
            raise ValueError(
                f"Could only assign {assigned_rows} rows to {split_name} while preserving train coverage for every drug and cell_line."
            )

    return pd.Series(split_assignments, index=frame.index, name="split")


def build_split_summary(split_frame, split_fractions):
    target_counts = compute_target_row_counts(len(split_frame), split_fractions)
    summary_rows = []

    for split_name in SPLIT_NAMES:
        subset = split_frame.loc[split_frame["split"] == split_name].copy()
        summary_rows.append(
            {
                "split": split_name,
                "target_rows": target_counts[split_name],
                "row_count": int(len(subset)),
                "requested_fraction": float(split_fractions[split_name]),
                "realized_fraction": float(len(subset) / len(split_frame)),
                "unique_drugs": int(subset["drug"].nunique()),
                "unique_cell_lines": int(subset["cell_line"].nunique()),
                "unique_organs": int(subset["organ"].nunique()),
            }
        )

    return pd.DataFrame(summary_rows)


def build_overlap_diagnostics(split_frame, split_mode):
    if split_mode == "drug_blind":
        entity_columns = ("drug", "condition_key", "cell_line")
    elif split_mode == "tumor_blind":
        entity_columns = ("cell_line", "condition_key", "drug")
    else:
        entity_columns = ("condition_key", "drug", "cell_line")

    split_sets = {
        split_name: {
            column: set(split_frame.loc[split_frame["split"] == split_name, column])
            for column in entity_columns
        }
        for split_name in SPLIT_NAMES
    }

    rows = []
    for column in entity_columns:
        for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
            rows.append(
                {
                    "entity": column,
                    "pair": f"{left_split}/{right_split}",
                    "overlap_count": len(split_sets[left_split][column] & split_sets[right_split][column]),
                }
            )

    return pd.DataFrame(rows)


def validate_split_assignments(split_frame, split_mode):
    if split_frame["condition_key"].duplicated().any():
        raise ValueError("condition_key values must remain unique after splitting.")

    if set(split_frame["split"]) != set(SPLIT_NAMES):
        raise ValueError("Every split must contain at least one row.")

    if split_mode == "drug_blind":
        split_sets = {
            split_name: set(split_frame.loc[split_frame["split"] == split_name, "drug"])
            for split_name in SPLIT_NAMES
        }
        for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
            if split_sets[left_split] & split_sets[right_split]:
                raise ValueError("drug_blind split leaked drugs across splits.")

    elif split_mode == "tumor_blind":
        split_sets = {
            split_name: set(split_frame.loc[split_frame["split"] == split_name, "cell_line"])
            for split_name in SPLIT_NAMES
        }
        for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
            if split_sets[left_split] & split_sets[right_split]:
                raise ValueError("tumor_blind split leaked cell lines across splits.")

    else:
        condition_key_sets = {
            split_name: set(split_frame.loc[split_frame["split"] == split_name, "condition_key"])
            for split_name in SPLIT_NAMES
        }
        for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
            if condition_key_sets[left_split] & condition_key_sets[right_split]:
                raise ValueError("mixed split leaked condition keys across splits.")

        train_frame = split_frame.loc[split_frame["split"] == "train"]
        if set(train_frame["drug"]) != set(split_frame["drug"]):
            raise ValueError("mixed split must retain every drug in train.")
        if set(train_frame["cell_line"]) != set(split_frame["cell_line"]):
            raise ValueError("mixed split must retain every cell line in train.")


def count_trainable_parameters(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def estimate_adamw_memory_gb(model):
    parameter_count = count_trainable_parameters(model)
    approximate_bytes = parameter_count * 4 * 4
    return float(approximate_bytes / (1024 ** 3))


def move_batch_to_device(batch, device):
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value
    return moved_batch


def _to_numpy_array(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def compute_mann_whitney_batch(predicted_expression, target_expression):
    predicted_expression_np = _to_numpy_array(predicted_expression)
    target_expression_np = _to_numpy_array(target_expression)
    mann_whitney_u_values = []
    mann_whitney_pvalues = []

    for predicted_row, target_row in zip(predicted_expression_np, target_expression_np):
        mann_whitney_result = mannwhitneyu(
            predicted_row,
            target_row,
            alternative="two-sided",
            method="asymptotic",
        )
        mann_whitney_u_values.append(float(mann_whitney_result.statistic))
        mann_whitney_pvalues.append(float(mann_whitney_result.pvalue))

    return (
        np.asarray(mann_whitney_u_values, dtype=np.float64),
        np.asarray(mann_whitney_pvalues, dtype=np.float64),
    )


def compute_topk_deg_metrics_batch(predicted_delta, target_delta, top_k=50):
    predicted_delta_np = _to_numpy_array(predicted_delta)
    target_delta_np = _to_numpy_array(target_delta)
    topk_match_counts = []
    topk_match_fractions = []
    signed_ndcg_at_k_values = []

    for predicted_row, target_row in zip(predicted_delta_np, target_delta_np):
        effective_k = min(int(top_k), int(predicted_row.shape[0]), int(target_row.shape[0]))
        if effective_k <= 0:
            topk_match_counts.append(0.0)
            topk_match_fractions.append(float("nan"))
            signed_ndcg_at_k_values.append(float("nan"))
            continue

        truth_topk = np.argsort(-np.abs(target_row), kind="stable")[:effective_k]
        predicted_topk = np.argsort(-np.abs(predicted_row), kind="stable")[:effective_k]
        predicted_topk_set = set(predicted_topk.tolist())
        match_count = len(set(truth_topk.tolist()) & predicted_topk_set)

        discounts = 1.0 / np.log2(np.arange(2, effective_k + 2, dtype=np.float64))
        ideal_gains = np.abs(target_row[truth_topk]).astype(np.float64)
        ideal_dcg = float(np.sum(ideal_gains * discounts))
        if ideal_dcg <= 0:
            signed_ndcg_at_k = 0.0
        else:
            signed_relevance = np.abs(target_row[predicted_topk]).astype(np.float64)
            sign_matches = (np.sign(predicted_row[predicted_topk]) == np.sign(target_row[predicted_topk])).astype(np.float64)
            signed_dcg = float(np.sum(signed_relevance * sign_matches * discounts))
            signed_ndcg_at_k = signed_dcg / ideal_dcg

        topk_match_counts.append(float(match_count))
        topk_match_fractions.append(float(match_count / effective_k))
        signed_ndcg_at_k_values.append(float(signed_ndcg_at_k))

    return (
        np.asarray(topk_match_counts, dtype=np.float64),
        np.asarray(topk_match_fractions, dtype=np.float64),
        np.asarray(signed_ndcg_at_k_values, dtype=np.float64),
    )


def evaluate_predictions_from_arrays(
    predicted_delta,
    baseline_expression,
    target_delta,
    split_name,
    gene_ids,
    metadata_df,
    max_inspection_rows=3,
    n_inspection_genes=5,
    mann_whitney_alpha=0.05,
    deg_top_k=50,
):
    predicted_delta_np = _to_numpy_array(predicted_delta).astype(np.float32, copy=False)
    baseline_expression_np = _to_numpy_array(baseline_expression).astype(np.float32, copy=False)
    target_delta_np = _to_numpy_array(target_delta).astype(np.float32, copy=False)

    if predicted_delta_np.shape != target_delta_np.shape:
        raise ValueError("predicted_delta and target_delta must have the same shape.")
    if baseline_expression_np.shape != target_delta_np.shape:
        raise ValueError("baseline_expression and target_delta must have the same shape.")

    metadata_frame = metadata_df.reset_index(drop=True).copy()
    n_rows = int(predicted_delta_np.shape[0])
    if len(metadata_frame) != n_rows:
        raise ValueError("metadata_df row count must align with prediction arrays.")
    if "dataset_index" not in metadata_frame.columns:
        metadata_frame.insert(0, "dataset_index", np.arange(n_rows, dtype=int))

    predicted_expression_np = baseline_expression_np + predicted_delta_np
    target_expression_np = baseline_expression_np + target_delta_np
    selected_gene_ids = list(gene_ids[:n_inspection_genes])

    per_sample_delta_mse = np.mean((predicted_delta_np - target_delta_np) ** 2, axis=1, dtype=np.float64)
    per_sample_delta_mae = np.mean(np.abs(predicted_delta_np - target_delta_np), axis=1, dtype=np.float64)
    per_sample_treated_mse = np.mean((predicted_expression_np - target_expression_np) ** 2, axis=1, dtype=np.float64)

    numerator = np.sum(predicted_expression_np * target_expression_np, axis=1, dtype=np.float64)
    predicted_norm = np.linalg.norm(predicted_expression_np, axis=1)
    target_norm = np.linalg.norm(target_expression_np, axis=1)
    cosine_denominator = np.clip(predicted_norm * target_norm, a_min=MIN_STANDARD_DEVIATION, a_max=None)
    per_sample_treated_cosine = numerator / cosine_denominator

    per_sample_mann_whitney_u, per_sample_mann_whitney_pvalue = compute_mann_whitney_batch(
        predicted_expression_np,
        target_expression_np,
    )
    (
        per_sample_top50_deg_match_count,
        per_sample_top50_deg_match_fraction,
        per_sample_signed_ndcg_at_50,
    ) = compute_topk_deg_metrics_batch(
        predicted_delta_np,
        target_delta_np,
        top_k=deg_top_k,
    )

    metrics = {
        "split": split_name,
        "n_samples": n_rows,
        "delta_mse": float(np.mean(per_sample_delta_mse)) if n_rows else float("nan"),
        "delta_mae": float(np.mean(per_sample_delta_mae)) if n_rows else float("nan"),
        "treated_cosine": float(np.mean(per_sample_treated_cosine)) if n_rows else float("nan"),
        "mann_whitney_u_median": float(np.median(per_sample_mann_whitney_u)) if n_rows else float("nan"),
        "mann_whitney_pvalue_mean": float(np.mean(per_sample_mann_whitney_pvalue)) if n_rows else float("nan"),
        "mann_whitney_pvalue_median": float(np.median(per_sample_mann_whitney_pvalue)) if n_rows else float("nan"),
        "mann_whitney_not_significant_fraction": float(np.mean(per_sample_mann_whitney_pvalue >= mann_whitney_alpha)) if n_rows else float("nan"),
        "top50_deg_match_count_mean": float(np.mean(per_sample_top50_deg_match_count)) if n_rows else float("nan"),
        "top50_deg_match_count_median": float(np.median(per_sample_top50_deg_match_count)) if n_rows else float("nan"),
        "top50_deg_match_fraction_mean": float(np.mean(per_sample_top50_deg_match_fraction)) if n_rows else float("nan"),
        "top50_deg_match_fraction_median": float(np.median(per_sample_top50_deg_match_fraction)) if n_rows else float("nan"),
        "signed_ndcg_at_50_mean": float(np.mean(per_sample_signed_ndcg_at_50)) if n_rows else float("nan"),
        "signed_ndcg_at_50_median": float(np.median(per_sample_signed_ndcg_at_50)) if n_rows else float("nan"),
    }

    inspection_rows = []
    for row_idx in range(min(max_inspection_rows, n_rows)):
        inspection_row = {
            "split": split_name,
            "condition_key": metadata_frame.loc[row_idx, "condition_key"],
            "cell_line": metadata_frame.loc[row_idx, "cell_line"],
            "drug": metadata_frame.loc[row_idx, "drug"],
            "concentration": float(metadata_frame.loc[row_idx, "concentration"]),
            "sample_delta_mse": float(per_sample_delta_mse[row_idx]),
            "sample_treated_cosine": float(per_sample_treated_cosine[row_idx]),
            "sample_mann_whitney_u": float(per_sample_mann_whitney_u[row_idx]),
            "sample_mann_whitney_pvalue": float(per_sample_mann_whitney_pvalue[row_idx]),
            "sample_top50_deg_match_count": float(per_sample_top50_deg_match_count[row_idx]),
            "sample_top50_deg_match_fraction": float(per_sample_top50_deg_match_fraction[row_idx]),
            "sample_signed_ndcg_at_50": float(per_sample_signed_ndcg_at_50[row_idx]),
        }
        for gene_offset, gene_id in enumerate(selected_gene_ids):
            inspection_row[f"pred_{gene_id}"] = float(predicted_expression_np[row_idx, gene_offset])
            inspection_row[f"target_{gene_id}"] = float(target_expression_np[row_idx, gene_offset])
        inspection_rows.append(inspection_row)

    prediction_detail_rows = []
    for row_idx in range(n_rows):
        metadata_row = metadata_frame.iloc[row_idx]
        prediction_detail_rows.append(
            {
                "split": split_name,
                "dataset_index": int(metadata_row["dataset_index"]),
                "condition_key": metadata_row["condition_key"],
                "cell_line": metadata_row["cell_line"],
                "cell_name": metadata_row["cell_name"],
                "organ": metadata_row["organ"],
                "drug": metadata_row["drug"],
                "concentration": float(metadata_row["concentration"]),
                "concentration_unit": metadata_row["concentration_unit"],
                "delta_mse": float(per_sample_delta_mse[row_idx]),
                "delta_mae": float(per_sample_delta_mae[row_idx]),
                "treated_mse": float(per_sample_treated_mse[row_idx]),
                "treated_cosine": float(per_sample_treated_cosine[row_idx]),
                "mann_whitney_u": float(per_sample_mann_whitney_u[row_idx]),
                "mann_whitney_pvalue": float(per_sample_mann_whitney_pvalue[row_idx]),
                "top50_deg_match_count": float(per_sample_top50_deg_match_count[row_idx]),
                "top50_deg_match_fraction": float(per_sample_top50_deg_match_fraction[row_idx]),
                "signed_ndcg_at_50": float(per_sample_signed_ndcg_at_50[row_idx]),
            }
        )

    return metrics, pd.DataFrame(inspection_rows), pd.DataFrame(prediction_detail_rows)


def evaluate_model_on_loader(
    model,
    loader,
    split_name,
    gene_ids,
    max_inspection_rows=3,
    n_inspection_genes=5,
    mann_whitney_alpha=0.05,
    deg_top_k=50,
):
    device = next(model.parameters()).device
    selected_gene_ids = list(gene_ids[:n_inspection_genes])
    total_rows = 0
    delta_mse_sum = 0.0
    delta_mae_sum = 0.0
    treated_cosine_sum = 0.0
    mann_whitney_u_values = []
    mann_whitney_pvalues = []
    top50_deg_match_counts = []
    top50_deg_match_fractions = []
    signed_ndcg_at_50_values = []
    inspection_rows = []
    prediction_detail_rows = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            predicted_delta = model(batch)
            target_delta = batch["target_delta"]
            predicted_expression = batch["baseline_expression"] + predicted_delta
            target_expression = batch["baseline_expression"] + target_delta

            per_sample_delta_mse = torch.mean((predicted_delta - target_delta) ** 2, dim=1)
            per_sample_delta_mae = torch.mean(torch.abs(predicted_delta - target_delta), dim=1)
            per_sample_treated_mse = torch.mean((predicted_expression - target_expression) ** 2, dim=1)
            per_sample_treated_cosine = F.cosine_similarity(predicted_expression, target_expression, dim=1)
            per_sample_mann_whitney_u, per_sample_mann_whitney_pvalue = compute_mann_whitney_batch(
                predicted_expression,
                target_expression,
            )
            (
                per_sample_top50_deg_match_count,
                per_sample_top50_deg_match_fraction,
                per_sample_signed_ndcg_at_50,
            ) = compute_topk_deg_metrics_batch(
                predicted_delta,
                target_delta,
                top_k=deg_top_k,
            )

            batch_rows = int(target_delta.shape[0])
            total_rows += batch_rows
            delta_mse_sum += float(per_sample_delta_mse.sum().item())
            delta_mae_sum += float(per_sample_delta_mae.sum().item())
            treated_cosine_sum += float(per_sample_treated_cosine.sum().item())
            mann_whitney_u_values.extend(per_sample_mann_whitney_u.tolist())
            mann_whitney_pvalues.extend(per_sample_mann_whitney_pvalue.tolist())
            top50_deg_match_counts.extend(per_sample_top50_deg_match_count.tolist())
            top50_deg_match_fractions.extend(per_sample_top50_deg_match_fraction.tolist())
            signed_ndcg_at_50_values.extend(per_sample_signed_ndcg_at_50.tolist())

            rows_needed = max(0, max_inspection_rows - len(inspection_rows))
            for row_idx in range(min(rows_needed, batch_rows)):
                inspection_row = {
                    "split": split_name,
                    "condition_key": batch["condition_key"][row_idx],
                    "cell_line": batch["cell_line"][row_idx],
                    "drug": batch["drug"][row_idx],
                    "concentration": float(batch["concentration"][row_idx].detach().cpu().item()),
                    "sample_delta_mse": float(per_sample_delta_mse[row_idx].detach().cpu().item()),
                    "sample_treated_cosine": float(per_sample_treated_cosine[row_idx].detach().cpu().item()),
                    "sample_mann_whitney_u": float(per_sample_mann_whitney_u[row_idx]),
                    "sample_mann_whitney_pvalue": float(per_sample_mann_whitney_pvalue[row_idx]),
                    "sample_top50_deg_match_count": float(per_sample_top50_deg_match_count[row_idx]),
                    "sample_top50_deg_match_fraction": float(per_sample_top50_deg_match_fraction[row_idx]),
                    "sample_signed_ndcg_at_50": float(per_sample_signed_ndcg_at_50[row_idx]),
                }
                for gene_offset, gene_id in enumerate(selected_gene_ids):
                    inspection_row[f"pred_{gene_id}"] = float(predicted_expression[row_idx, gene_offset].detach().cpu().item())
                    inspection_row[f"target_{gene_id}"] = float(target_expression[row_idx, gene_offset].detach().cpu().item())
                inspection_rows.append(inspection_row)

            dataset_indices = batch["dataset_index"].detach().cpu().numpy()
            concentrations = batch["concentration"].detach().cpu().numpy()
            delta_mse_values = per_sample_delta_mse.detach().cpu().numpy()
            delta_mae_values = per_sample_delta_mae.detach().cpu().numpy()
            treated_mse_values = per_sample_treated_mse.detach().cpu().numpy()
            treated_cosine_values = per_sample_treated_cosine.detach().cpu().numpy()
            for row_idx in range(batch_rows):
                prediction_detail_rows.append(
                    {
                        "split": split_name,
                        "dataset_index": int(dataset_indices[row_idx]),
                        "condition_key": batch["condition_key"][row_idx],
                        "cell_line": batch["cell_line"][row_idx],
                        "cell_name": batch["cell_name"][row_idx],
                        "organ": batch["organ"][row_idx],
                        "drug": batch["drug"][row_idx],
                        "concentration": float(concentrations[row_idx]),
                        "concentration_unit": batch["concentration_unit"][row_idx],
                        "delta_mse": float(delta_mse_values[row_idx]),
                        "delta_mae": float(delta_mae_values[row_idx]),
                        "treated_mse": float(treated_mse_values[row_idx]),
                        "treated_cosine": float(treated_cosine_values[row_idx]),
                        "mann_whitney_u": float(per_sample_mann_whitney_u[row_idx]),
                        "mann_whitney_pvalue": float(per_sample_mann_whitney_pvalue[row_idx]),
                        "top50_deg_match_count": float(per_sample_top50_deg_match_count[row_idx]),
                        "top50_deg_match_fraction": float(per_sample_top50_deg_match_fraction[row_idx]),
                        "signed_ndcg_at_50": float(per_sample_signed_ndcg_at_50[row_idx]),
                    }
                )

    mann_whitney_u_values = np.asarray(mann_whitney_u_values, dtype=np.float64)
    mann_whitney_pvalues = np.asarray(mann_whitney_pvalues, dtype=np.float64)
    top50_deg_match_counts = np.asarray(top50_deg_match_counts, dtype=np.float64)
    top50_deg_match_fractions = np.asarray(top50_deg_match_fractions, dtype=np.float64)
    signed_ndcg_at_50_values = np.asarray(signed_ndcg_at_50_values, dtype=np.float64)

    metrics = {
        "split": split_name,
        "n_samples": int(total_rows),
        "delta_mse": float(delta_mse_sum / max(total_rows, 1)),
        "delta_mae": float(delta_mae_sum / max(total_rows, 1)),
        "treated_cosine": float(treated_cosine_sum / max(total_rows, 1)),
        "mann_whitney_u_median": float(np.median(mann_whitney_u_values)) if total_rows else float("nan"),
        "mann_whitney_pvalue_mean": float(np.mean(mann_whitney_pvalues)) if total_rows else float("nan"),
        "mann_whitney_pvalue_median": float(np.median(mann_whitney_pvalues)) if total_rows else float("nan"),
        "mann_whitney_not_significant_fraction": float(np.mean(mann_whitney_pvalues >= mann_whitney_alpha)) if total_rows else float("nan"),
        "top50_deg_match_count_mean": float(np.mean(top50_deg_match_counts)) if total_rows else float("nan"),
        "top50_deg_match_count_median": float(np.median(top50_deg_match_counts)) if total_rows else float("nan"),
        "top50_deg_match_fraction_mean": float(np.mean(top50_deg_match_fractions)) if total_rows else float("nan"),
        "top50_deg_match_fraction_median": float(np.median(top50_deg_match_fractions)) if total_rows else float("nan"),
        "signed_ndcg_at_50_mean": float(np.mean(signed_ndcg_at_50_values)) if total_rows else float("nan"),
        "signed_ndcg_at_50_median": float(np.median(signed_ndcg_at_50_values)) if total_rows else float("nan"),
    }
    inspection_df = pd.DataFrame(inspection_rows)
    prediction_details_df = pd.DataFrame(prediction_detail_rows)
    return metrics, inspection_df, prediction_details_df


def load_lightning_metrics_table(metrics_csv_path):
    metrics_csv_path = Path(metrics_csv_path)
    if not metrics_csv_path.exists():
        raise FileNotFoundError(f"Could not find Lightning metrics file: {metrics_csv_path}")

    metrics_df = pd.read_csv(metrics_csv_path)
    if metrics_df.empty:
        raise ValueError("Lightning metrics.csv is empty.")
    if "epoch" not in metrics_df.columns:
        raise ValueError("Lightning metrics.csv does not contain an epoch column.")
    return metrics_df


def build_loss_history_table(metrics_df):
    history_frames = []
    for split_name, loss_column in (("train", "train_loss"), ("val", "val_loss")):
        if loss_column not in metrics_df.columns:
            continue
        subset = metrics_df.loc[metrics_df[loss_column].notna(), ["epoch", "step", loss_column]].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["epoch", "step"], kind="stable")
        subset = subset.groupby("epoch", as_index=False)[loss_column].last()
        subset = subset.rename(columns={loss_column: "loss"})
        subset["split"] = split_name
        history_frames.append(subset)

    if not history_frames:
        raise ValueError("Could not extract any epoch-level train/val loss values from Lightning metrics.csv.")

    loss_history_df = pd.concat(history_frames, ignore_index=True)
    loss_history_df["epoch"] = loss_history_df["epoch"].astype(int)
    return loss_history_df.sort_values(["epoch", "split"], ignore_index=True)


def sample_prediction_details(prediction_details_df, sample_count, sample_strategy, random_seed):
    if prediction_details_df.empty:
        raise ValueError("prediction_details_df is empty; there are no rows to sample.")
    if sample_strategy != "seeded_random":
        raise ValueError(f"Unsupported sample strategy: {sample_strategy}")

    effective_sample_count = min(int(sample_count), int(len(prediction_details_df)))
    sampled_df = prediction_details_df.sample(
        n=effective_sample_count,
        replace=False,
        random_state=int(random_seed),
    )
    sampled_df = sampled_df.sort_values(["dataset_index", "condition_key"], kind="stable").reset_index(drop=True)
    sampled_df["pair_index"] = np.arange(1, len(sampled_df) + 1, dtype=int)
    return sampled_df


def build_prediction_pair_embedding(model, dataset, sampled_prediction_details_df, embedding_method, n_components, random_seed):
    if embedding_method != "pca":
        raise ValueError(f"Unsupported embedding method: {embedding_method}")
    if sampled_prediction_details_df.empty:
        raise ValueError("No sampled prediction details were provided for embedding.")

    device = next(model.parameters()).device
    selected_examples = [dataset[int(dataset_index)] for dataset_index in sampled_prediction_details_df["dataset_index"].tolist()]
    model_input_batch = {
        "input_features": torch.stack([example["input_features"] for example in selected_examples], dim=0).to(device),
        "gene_features": torch.stack([example["gene_features"] for example in selected_examples], dim=0).to(device),
        "drug_features": torch.stack([example["drug_features"] for example in selected_examples], dim=0).to(device),
        "dose_feature": torch.stack([example["dose_feature"] for example in selected_examples], dim=0).to(device),
    }
    baseline_expression = torch.stack([example["baseline_expression"] for example in selected_examples], dim=0).to(device)
    target_delta = torch.stack([example["target_delta"] for example in selected_examples], dim=0).to(device)

    model.eval()
    with torch.no_grad():
        predicted_delta = model(model_input_batch)
        predicted_expression = baseline_expression + predicted_delta
        actual_expression = baseline_expression + target_delta
    return build_prediction_pair_embedding_from_arrays(
        predicted_expression=predicted_expression.detach().cpu().numpy(),
        actual_expression=actual_expression.detach().cpu().numpy(),
        sampled_prediction_details_df=sampled_prediction_details_df,
        embedding_method=embedding_method,
        n_components=n_components,
        random_seed=random_seed,
    )


def build_prediction_pair_embedding_from_arrays(
    predicted_expression,
    actual_expression,
    sampled_prediction_details_df,
    embedding_method,
    n_components,
    random_seed,
):
    if embedding_method != "pca":
        raise ValueError(f"Unsupported embedding method: {embedding_method}")
    if sampled_prediction_details_df.empty:
        raise ValueError("No sampled prediction details were provided for embedding.")

    predicted_expression_np = _to_numpy_array(predicted_expression).astype(np.float32, copy=False)
    actual_expression_np = _to_numpy_array(actual_expression).astype(np.float32, copy=False)
    if predicted_expression_np.shape != actual_expression_np.shape:
        raise ValueError("predicted_expression and actual_expression must have the same shape.")

    dataset_indices = sampled_prediction_details_df["dataset_index"].to_numpy(np.int64)
    sampled_predicted_expression = predicted_expression_np[dataset_indices]
    sampled_actual_expression = actual_expression_np[dataset_indices]

    combined_expression_np = np.concatenate([sampled_actual_expression, sampled_predicted_expression], axis=0)
    embedding_model = PCA(n_components=int(n_components), svd_solver="full", random_state=int(random_seed))
    embedded_points = embedding_model.fit_transform(combined_expression_np)

    n_pairs = len(sampled_prediction_details_df)
    actual_points = embedded_points[:n_pairs]
    predicted_points = embedded_points[n_pairs:]
    explained_variance_ratio = embedding_model.explained_variance_ratio_

    plot_rows = []
    pair_summary_rows = []
    for row_idx, metadata_row in sampled_prediction_details_df.reset_index(drop=True).iterrows():
        actual_point = actual_points[row_idx]
        predicted_point = predicted_points[row_idx]
        pair_distance_2d = float(np.linalg.norm(actual_point - predicted_point))

        for point_kind, point_values in (("actual", actual_point), ("predicted", predicted_point)):
            plot_rows.append(
                {
                    "pair_index": int(metadata_row["pair_index"]),
                    "point_kind": point_kind,
                    "embedding_1": float(point_values[0]),
                    "embedding_2": float(point_values[1]),
                    "condition_key": metadata_row["condition_key"],
                    "cell_line": metadata_row["cell_line"],
                    "drug": metadata_row["drug"],
                    "concentration": float(metadata_row["concentration"]),
                }
            )

        pair_summary_rows.append(
            {
                "pair_index": int(metadata_row["pair_index"]),
                "condition_key": metadata_row["condition_key"],
                "cell_line": metadata_row["cell_line"],
                "drug": metadata_row["drug"],
                "concentration": float(metadata_row["concentration"]),
                "concentration_unit": metadata_row["concentration_unit"],
                "treated_mse": float(metadata_row["treated_mse"]),
                "delta_mse": float(metadata_row["delta_mse"]),
                "treated_cosine": float(metadata_row["treated_cosine"]),
                "mann_whitney_u": float(metadata_row["mann_whitney_u"]),
                "mann_whitney_pvalue": float(metadata_row["mann_whitney_pvalue"]),
                "top50_deg_match_count": float(metadata_row["top50_deg_match_count"]),
                "top50_deg_match_fraction": float(metadata_row["top50_deg_match_fraction"]),
                "signed_ndcg_at_50": float(metadata_row["signed_ndcg_at_50"]),
                "pair_distance_2d": pair_distance_2d,
            }
        )

    plot_df = pd.DataFrame(plot_rows)
    pair_summary_df = pd.DataFrame(pair_summary_rows).sort_values(
        ["pair_distance_2d", "treated_mse"],
        ascending=[False, False],
        ignore_index=True,
    )
    return plot_df, pair_summary_df, explained_variance_ratio
