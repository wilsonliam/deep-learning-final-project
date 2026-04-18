"""Builds Machine_Learning/9_model_comparison.ipynb from inline cell definitions.

Run once to (re)generate the notebook. Keeping the cells as Python strings in this
script rather than hand-editing JSON keeps diffs reviewable.
"""
import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = []

CELLS.append(md("""# Final Model Comparison

This notebook compares the four drug-response models trained in notebooks 4, 6, 7, and 8 against
each other on a common footing, and additionally against the λ=0 plate-adversary ablation from
notebook 8:

| Tag | Source notebook | Tensor artifacts | Architecture | Confounder adversary |
| --- | --- | --- | --- | --- |
| `mlp` | `4_Model_implementatin.ipynb` | `Tahoe100M_tensor_artifacts_L1000` | Branched MLP (gene+drug+dose → 977 genes) | none |
| `random_forest` | `6_random_forest_model.ipynb` | `Tahoe100M_tensor_artifacts_L1000` | `sklearn.ensemble.RandomForestRegressor` | none |
| `adae_cell_line` | `7_adae_model.ipynb` | `Tahoe100M_tensor_artifacts_L1000` | AD-AE (ctx+drug encoders, concat fuser, GRL) | cell line |
| `adae_plate` | `8_adae_plate.ipynb` | `Tahoe100M_tensor_artifacts_L1000_plate` | AD-AE (FiLM fuser, drug-conditioned adversary) | plate |
| `adae_plate_ablation_lambda0` | `8_adae_plate.ipynb` (ablation) | `Tahoe100M_tensor_artifacts_L1000_plate` | same AD-AE as `adae_plate` with λ pinned to 0 | plate (disabled) |

The five model checkpoints live under `artifacts/` and are consumed read-only here — no training
happens in this notebook. We rebuild each model's train/val/test split from its original tensor
bundle (same `drug_blind` + seed), then reload the trained weights and run `evaluate_model_on_loader`
/ `evaluate_sklearn_predictions` to get the identical metric panel every other notebook reports.

Because the L1000 and L1000_plate bundles contain different numbers of rows per drug, the
`drug_blind` split is NOT bit-identical across the two tensor sets. Models trained on the same
tensor set (MLP / RF / ADAE-cell-line on L1000; ADAE-plate + ablation on L1000_plate) share a
test set; cross-tensor comparisons should therefore be read with that caveat in mind. Every
metric below is reported with the underlying sample count so the reader can spot the split
discrepancy directly.
"""))

CELLS.append(md("## Configuration\n"))

CELLS.append(code('''from pathlib import Path

SPLIT_MODE = "drug_blind"
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
BATCH_SIZE = 512
RANDOM_SEED = 42
NUM_WORKERS = 0
PREPROCESS_BATCH_SIZE = 1024
TARGET_MODE = "delta"

L1000_TENSOR_DIR = Path("data/Tahoe100M_tensor_artifacts_L1000")
L1000_PLATE_TENSOR_DIR = Path("data/Tahoe100M_tensor_artifacts_L1000_plate")

MODEL_CHECKPOINT_PATHS = {
    "mlp": Path(
        "artifacts/lightning/linear_regression/drug_blind_mlp_delta/"
        "checkpoints/epoch=epoch=00-val_loss=val_loss=591.767273.ckpt"
    ),
    "adae_cell_line": Path(
        "artifacts/lightning/adae/drug_blind_adae_delta/"
        "checkpoints/epoch=epoch=02-val_treated_cosine=val_treated_cosine=0.981929.ckpt"
    ),
    "adae_plate": Path(
        "artifacts/lightning/adae_plate/drug_blind_adae_plate_delta/"
        "checkpoints/epoch=epoch=01-val_delta_pearson=val_delta_pearson=0.432664.ckpt"
    ),
    "adae_plate_ablation_lambda0": Path(
        "artifacts/lightning/adae_plate/drug_blind_adae_plate_delta_ablation_lambda0/"
        "checkpoints/epoch=epoch=02-val_delta_pearson=val_delta_pearson=0.400924.ckpt"
    ),
}
RANDOM_FOREST_JOBLIB_PATH = Path(
    "artifacts/sklearn/random_forest/drug_blind_random_forest_delta/random_forest.joblib"
)
MODEL_METRICS_CSV_PATHS = {
    "mlp": Path("artifacts/lightning/linear_regression/drug_blind_mlp_delta/version_1/metrics.csv"),
    "adae_cell_line": Path("artifacts/lightning/adae/drug_blind_adae_delta/version_0/metrics.csv"),
    "adae_plate": Path("artifacts/lightning/adae_plate/drug_blind_adae_plate_delta/version_4/metrics.csv"),
    "adae_plate_ablation_lambda0": Path(
        "artifacts/lightning/adae_plate/drug_blind_adae_plate_delta_ablation_lambda0/version_2/metrics.csv"
    ),
}

L1000_TENSOR_KEY = "L1000"
L1000_PLATE_TENSOR_KEY = "L1000_plate"

MODEL_TO_TENSOR_KEY = {
    "mlp": L1000_TENSOR_KEY,
    "random_forest": L1000_TENSOR_KEY,
    "adae_cell_line": L1000_TENSOR_KEY,
    "adae_plate": L1000_PLATE_TENSOR_KEY,
    "adae_plate_ablation_lambda0": L1000_PLATE_TENSOR_KEY,
}
MODEL_DISPLAY_ORDER = [
    "mlp",
    "random_forest",
    "adae_cell_line",
    "adae_plate",
    "adae_plate_ablation_lambda0",
]
PLATE_CONFOUNDER_COLUMN = "plate"

MODEL_CHECKPOINT_PATHS
'''))

CELLS.append(md("## Imports and Project Root\n"))

CELLS.append(code('''import copy
import sys
import warnings
from pathlib import Path

import joblib
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display

for candidate_root in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    if (candidate_root / "pyproject.toml").exists() and (candidate_root / "Machine_Learning" / "ml_pipeline").exists():
        candidate_root_str = str(candidate_root)
        if candidate_root_str not in sys.path:
            sys.path.insert(0, candidate_root_str)
        break
else:
    raise ModuleNotFoundError(
        "Could not resolve the project root needed to import Machine_Learning.ml_pipeline."
    )

from Machine_Learning.ml_pipeline.data import DrugResponseDataModule, TrainingPreprocessor
from Machine_Learning.ml_pipeline.models import (
    ADAEDrugResponseModule,
    MLPDrugResponseModule,
    probe_latent_plate_accuracy,
)
from Machine_Learning.ml_pipeline.utils import (
    SPLIT_NAMES,
    assign_group_blind_splits,
    build_overlap_diagnostics,
    build_rf_training_arrays,
    build_split_summary,
    compute_condition_retrieval,
    compute_null_baselines_from_loaders,
    evaluate_model_on_loader,
    evaluate_sklearn_predictions,
    load_cached_cell_line_metadata,
    load_lightning_metrics_table,
    move_batch_to_device,
    resolve_project_path,
    resolve_retrieval_label_column,
    resolve_target_gene_indices,
    validate_split_assignments,
)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

L.seed_everything(RANDOM_SEED, workers=True)
'''))

CELLS.append(md("""## Build Matching Data Pipelines

Each tensor bundle gets its own `examples_df` + `drug_blind` split + `TrainingPreprocessor`.
We evaluate every model against *its own* training-time split so the test row distribution
matches what the checkpoint saw during training. Sample counts per split are printed below
so the two pipelines can be compared row-for-row.
"""))

CELLS.append(code('''def load_tensor_bundles(tensor_artifacts_dir, expect_plate):
    tensor_artifacts_dir = resolve_project_path(tensor_artifacts_dir)
    dmso_bundle = torch.load(tensor_artifacts_dir / "dmso_baselines.pt", map_location="cpu")
    treatment_bundle = torch.load(tensor_artifacts_dir / "treatment_expressions.pt", map_location="cpu")
    fingerprint_bundle = torch.load(tensor_artifacts_dir / "morgan_fingerprints.pt", map_location="cpu")

    if expect_plate and ("plates" not in treatment_bundle or "plate_to_index" not in treatment_bundle):
        raise KeyError(
            f"Plate-resolved tensors expected at {tensor_artifacts_dir}, but bundle has no "
            "'plates' / 'plate_to_index'. Rebuild via 3.5_Make_plate_tensors.ipynb."
        )
    return dmso_bundle, treatment_bundle, fingerprint_bundle, tensor_artifacts_dir


def build_examples_df(dmso_bundle, treatment_bundle, fingerprint_bundle, include_plate):
    examples = {
        "condition_key": treatment_bundle["condition_keys"],
        "cell_line": treatment_bundle["cell_lines"],
        "file_name": treatment_bundle["file_names"],
        "drug": treatment_bundle["drug_names"],
        "concentration": treatment_bundle["concentrations"].cpu().numpy().astype(np.float32),
        "concentration_unit": treatment_bundle["concentration_units"],
        "target_index": np.arange(len(treatment_bundle["condition_keys"]), dtype=np.int64),
    }
    if include_plate:
        examples["plate"] = treatment_bundle["plates"]
    examples_df = pd.DataFrame(examples)
    examples_df["baseline_index"] = examples_df["cell_line"].map(dmso_bundle["cell_line_to_index"])
    examples_df["fingerprint_index"] = examples_df["drug"].map(fingerprint_bundle["drug_to_index"])

    if examples_df["condition_key"].duplicated().any():
        raise ValueError("condition_key values must be unique in the treatment bundle.")
    if examples_df["baseline_index"].isna().any():
        raise ValueError("Some treatment rows do not resolve to a DMSO baseline index.")
    if examples_df["fingerprint_index"].isna().any():
        raise ValueError("Some treatment rows do not resolve to a Morgan fingerprint index.")
    if include_plate and examples_df["plate"].isna().any():
        raise ValueError("Some treatment rows have a missing plate label.")

    cell_line_metadata_df, _ = load_cached_cell_line_metadata()
    tensor_cell_lines = set(dmso_bundle["cell_lines"])
    matched_cell_line_metadata_df = cell_line_metadata_df.loc[
        cell_line_metadata_df["cell_line"].isin(tensor_cell_lines)
    ].copy()
    missing_cell_line_metadata = sorted(tensor_cell_lines - set(matched_cell_line_metadata_df["cell_line"]))
    if missing_cell_line_metadata:
        raise ValueError(f"Missing cell-line metadata for: {missing_cell_line_metadata}")

    examples_df = examples_df.merge(
        matched_cell_line_metadata_df,
        on="cell_line",
        how="left",
        validate="many_to_one",
    )
    if examples_df[["cell_name", "organ"]].isna().any().any():
        raise ValueError("Some treatment rows do not resolve to cell-line metadata.")

    examples_df[["baseline_index", "fingerprint_index", "target_index"]] = examples_df[[
        "baseline_index",
        "fingerprint_index",
        "target_index",
    ]].astype(int)
    return examples_df


def prepare_pipeline(tensor_artifacts_dir, include_plate):
    dmso_bundle, treatment_bundle, fingerprint_bundle, resolved_dir = load_tensor_bundles(
        tensor_artifacts_dir,
        expect_plate=include_plate,
    )
    input_gene_ids = [str(gene_id) for gene_id in dmso_bundle["gene_ids"]]
    target_gene_ids = [str(gene_id) for gene_id in treatment_bundle["gene_ids"]]
    target_gene_indices = resolve_target_gene_indices(input_gene_ids, target_gene_ids)

    examples_df = build_examples_df(dmso_bundle, treatment_bundle, fingerprint_bundle, include_plate=include_plate)
    split_assignments = assign_group_blind_splits(
        examples_df,
        group_col="drug",
        split_fractions=SPLIT_FRACTIONS,
        seed=RANDOM_SEED,
    )
    split_examples_df = examples_df.copy()
    split_examples_df["split"] = split_assignments.to_numpy()
    validate_split_assignments(split_examples_df, SPLIT_MODE)

    train_examples_df = split_examples_df.loc[split_examples_df["split"] == "train"].reset_index(drop=True)
    val_examples_df = split_examples_df.loc[split_examples_df["split"] == "val"].reset_index(drop=True)
    test_examples_df = split_examples_df.loc[split_examples_df["split"] == "test"].reset_index(drop=True)

    preprocessor = TrainingPreprocessor(
        train_examples_df=train_examples_df,
        dmso_bundle=dmso_bundle,
        treatment_bundle=treatment_bundle,
        fingerprint_bundle=fingerprint_bundle,
        target_gene_indices=target_gene_indices,
        target_mode=TARGET_MODE,
        batch_size=PREPROCESS_BATCH_SIZE,
    ).fit()

    confounder_column = None
    confounder_to_index = None
    if include_plate:
        train_plates = sorted(train_examples_df[PLATE_CONFOUNDER_COLUMN].unique().tolist())
        plate_to_index = {plate: idx for idx, plate in enumerate(train_plates)}
        val_before = len(val_examples_df)
        test_before = len(test_examples_df)
        val_examples_df = val_examples_df.loc[
            val_examples_df[PLATE_CONFOUNDER_COLUMN].isin(plate_to_index)
        ].reset_index(drop=True)
        test_examples_df = test_examples_df.loc[
            test_examples_df[PLATE_CONFOUNDER_COLUMN].isin(plate_to_index)
        ].reset_index(drop=True)
        if len(val_examples_df) < val_before or len(test_examples_df) < test_before:
            warnings.warn(
                f"Dropped val ({val_before - len(val_examples_df)}) / test "
                f"({test_before - len(test_examples_df)}) rows on plates unseen in train."
            )
        confounder_column = PLATE_CONFOUNDER_COLUMN
        confounder_to_index = plate_to_index

    data_module = DrugResponseDataModule(
        train_examples_df=train_examples_df,
        val_examples_df=val_examples_df,
        test_examples_df=test_examples_df,
        preprocessor=preprocessor,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
        confounder_column=confounder_column,
        confounder_to_index=confounder_to_index,
    )
    data_module.setup()

    split_summary_df = build_split_summary(split_examples_df, SPLIT_FRACTIONS)
    overlap_diagnostics_df = build_overlap_diagnostics(split_examples_df, SPLIT_MODE)
    return {
        "tensor_artifacts_dir": resolved_dir,
        "dmso_bundle": dmso_bundle,
        "treatment_bundle": treatment_bundle,
        "fingerprint_bundle": fingerprint_bundle,
        "input_gene_ids": input_gene_ids,
        "target_gene_ids": target_gene_ids,
        "target_gene_indices": target_gene_indices,
        "examples_df": examples_df,
        "split_examples_df": split_examples_df,
        "train_examples_df": train_examples_df,
        "val_examples_df": val_examples_df,
        "test_examples_df": test_examples_df,
        "preprocessor": preprocessor,
        "data_module": data_module,
        "split_summary_df": split_summary_df,
        "overlap_diagnostics_df": overlap_diagnostics_df,
        "confounder_column": confounder_column,
        "confounder_to_index": confounder_to_index,
    }


pipelines = {
    L1000_TENSOR_KEY: prepare_pipeline(L1000_TENSOR_DIR, include_plate=False),
    L1000_PLATE_TENSOR_KEY: prepare_pipeline(L1000_PLATE_TENSOR_DIR, include_plate=True),
}

pipeline_summary_rows = []
for tensor_key, pipeline in pipelines.items():
    pipeline_summary_rows.append(
        {
            "tensor_key": tensor_key,
            "tensor_artifacts_dir": str(pipeline["tensor_artifacts_dir"]),
            "n_examples": int(len(pipeline["examples_df"])),
            "n_train": int(len(pipeline["train_examples_df"])),
            "n_val": int(len(pipeline["val_examples_df"])),
            "n_test": int(len(pipeline["test_examples_df"])),
            "n_unique_drugs": int(pipeline["examples_df"]["drug"].nunique()),
            "n_unique_cell_lines": int(pipeline["examples_df"]["cell_line"].nunique()),
            "n_unique_plates": (
                int(pipeline["examples_df"]["plate"].nunique())
                if "plate" in pipeline["examples_df"].columns
                else np.nan
            ),
            "target_gene_dim": int(pipeline["preprocessor"].target_gene_dim),
        }
    )
pipeline_summary_df = pd.DataFrame(pipeline_summary_rows)
display(pipeline_summary_df)
'''))

CELLS.append(md("""## Load Trained Models

Every Lightning model is reconstructed via `load_from_checkpoint`, passing the freshly fit
preprocessor's `delta_mean` / `delta_std` as kwargs. The checkpoint's saved `state_dict` then
overwrites the buffers with the training-time statistics, so predictions are scaled identically
to what the original notebook produced. The random forest is loaded via `joblib`.
"""))

CELLS.append(code('''def load_lightning_adae_model(checkpoint_path, preprocessor):
    checkpoint_path = resolve_project_path(checkpoint_path)
    model = ADAEDrugResponseModule.load_from_checkpoint(
        str(checkpoint_path),
        delta_mean=preprocessor.delta_mean,
        delta_std=preprocessor.delta_std,
        map_location="cpu",
    )
    model.eval()
    return model


def load_lightning_mlp_model(checkpoint_path, preprocessor):
    checkpoint_path = resolve_project_path(checkpoint_path)
    model = MLPDrugResponseModule.load_from_checkpoint(
        str(checkpoint_path),
        delta_mean=preprocessor.delta_mean,
        delta_std=preprocessor.delta_std,
        map_location="cpu",
    )
    model.eval()
    return model


l1000_preprocessor = pipelines[L1000_TENSOR_KEY]["preprocessor"]
l1000_plate_preprocessor = pipelines[L1000_PLATE_TENSOR_KEY]["preprocessor"]

loaded_models = {
    "mlp": load_lightning_mlp_model(MODEL_CHECKPOINT_PATHS["mlp"], l1000_preprocessor),
    "adae_cell_line": load_lightning_adae_model(
        MODEL_CHECKPOINT_PATHS["adae_cell_line"], l1000_preprocessor
    ),
    "adae_plate": load_lightning_adae_model(
        MODEL_CHECKPOINT_PATHS["adae_plate"], l1000_plate_preprocessor
    ),
    "adae_plate_ablation_lambda0": load_lightning_adae_model(
        MODEL_CHECKPOINT_PATHS["adae_plate_ablation_lambda0"], l1000_plate_preprocessor
    ),
}

rf_estimator = joblib.load(resolve_project_path(RANDOM_FOREST_JOBLIB_PATH))

loaded_model_summary_rows = []
for model_name in MODEL_DISPLAY_ORDER:
    tensor_key = MODEL_TO_TENSOR_KEY[model_name]
    if model_name == "random_forest":
        loaded_model_summary_rows.append(
            {
                "model_name": model_name,
                "tensor_key": tensor_key,
                "artifact": str(resolve_project_path(RANDOM_FOREST_JOBLIB_PATH)),
                "architecture": "sklearn_random_forest",
                "n_parameters": int(getattr(rf_estimator, "n_estimators", 0)),
                "device": "cpu",
            }
        )
        continue
    module = loaded_models[model_name]
    parameter_count = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    loaded_model_summary_rows.append(
        {
            "model_name": model_name,
            "tensor_key": tensor_key,
            "artifact": str(resolve_project_path(MODEL_CHECKPOINT_PATHS[model_name])),
            "architecture": module.__class__.__name__,
            "n_parameters": parameter_count,
            "device": str(next(module.parameters()).device),
        }
    )
loaded_model_summary_df = pd.DataFrame(loaded_model_summary_rows)
display(loaded_model_summary_df)
'''))

CELLS.append(md("""## Evaluate Every Model on Train / Val / Test

For the Lightning models we loop over the split loaders and call `evaluate_model_on_loader` —
the same helper used in notebooks 4, 7, and 8. The random forest uses `evaluate_sklearn_predictions`,
which consumes the same RF assembly helpers as notebook 6. We also collect per-row prediction detail
frames keyed by `(model_name, split)` so the downstream comparisons can group by cell line, plate,
drug, etc. without re-running inference.
"""))

CELLS.append(code('''def evaluate_lightning_model(model_name, model, pipeline):
    data_module = pipeline["data_module"]
    evaluation_loaders = {
        "train": data_module.train_eval_dataloader(),
        "val": data_module.val_dataloader(),
        "test": data_module.test_dataloader(),
    }
    metrics_rows = []
    prediction_details = {}
    prediction_delta_arrays = {}

    retrieval_label_column = resolve_retrieval_label_column(SPLIT_MODE)
    model_device = next(model.parameters()).device
    target_gene_ids = pipeline["target_gene_ids"]

    model.eval()
    for split_name, loader in evaluation_loaders.items():
        metrics_row, _, prediction_details_df = evaluate_model_on_loader(
            model=model,
            loader=loader,
            split_name=split_name,
            gene_ids=target_gene_ids,
        )
        metrics_row["model_name"] = model_name
        metrics_rows.append(metrics_row)
        prediction_details[split_name] = prediction_details_df

        predicted_delta_chunks = []
        target_delta_chunks = []
        label_chunks = []
        with torch.no_grad():
            for batch in loader:
                batch_on_device = move_batch_to_device(batch, model_device)
                predicted_delta_chunks.append(model(batch_on_device).detach().cpu().numpy())
                target_delta_chunks.append(batch_on_device["target_delta"].detach().cpu().numpy())
                label_values = batch[retrieval_label_column]
                if isinstance(label_values, (list, tuple)):
                    label_chunks.extend(str(value) for value in label_values)
                else:
                    label_chunks.extend(str(value) for value in list(label_values))
        prediction_delta_arrays[split_name] = (
            np.concatenate(predicted_delta_chunks, axis=0)
            if predicted_delta_chunks
            else np.zeros((0, len(target_gene_ids)), dtype=np.float32),
            np.concatenate(target_delta_chunks, axis=0)
            if target_delta_chunks
            else np.zeros((0, len(target_gene_ids)), dtype=np.float32),
            np.asarray(label_chunks, dtype=object),
        )

    return metrics_rows, prediction_details, prediction_delta_arrays


def evaluate_random_forest(pipeline, rf_estimator):
    preprocessor = pipeline["preprocessor"]
    train_examples_df = pipeline["train_examples_df"]
    val_examples_df = pipeline["val_examples_df"]
    test_examples_df = pipeline["test_examples_df"]

    X_train, y_train, baseline_train = build_rf_training_arrays(preprocessor, train_examples_df)
    X_val, y_val, baseline_val = build_rf_training_arrays(preprocessor, val_examples_df)
    X_test, y_test, baseline_test = build_rf_training_arrays(preprocessor, test_examples_df)

    split_arrays = {
        "train": (X_train, y_train, baseline_train, train_examples_df),
        "val": (X_val, y_val, baseline_val, val_examples_df),
        "test": (X_test, y_test, baseline_test, test_examples_df),
    }
    predicted_delta_by_split = {name: rf_estimator.predict(X) for name, (X, *_rest) in split_arrays.items()}

    metrics_rows = []
    prediction_details = {}
    prediction_delta_arrays = {}
    retrieval_label_column = resolve_retrieval_label_column(SPLIT_MODE)

    for split_name, (_, y_split, baseline_split, metadata_split) in split_arrays.items():
        metrics_row, _, prediction_details_df = evaluate_sklearn_predictions(
            predicted_delta_np=predicted_delta_by_split[split_name],
            target_delta_np=y_split,
            baseline_np=baseline_split,
            metadata_df=metadata_split,
            split_name=split_name,
            gene_ids=pipeline["target_gene_ids"],
        )
        metrics_row["model_name"] = "random_forest"
        metrics_rows.append(metrics_row)
        prediction_details[split_name] = prediction_details_df
        prediction_delta_arrays[split_name] = (
            predicted_delta_by_split[split_name].astype(np.float32, copy=False),
            y_split.astype(np.float32, copy=False),
            metadata_split[retrieval_label_column].astype(str).to_numpy(),
        )

    return metrics_rows, prediction_details, prediction_delta_arrays


all_metrics_rows = []
prediction_details_by_model = {}
prediction_delta_arrays_by_model = {}

for model_name in MODEL_DISPLAY_ORDER:
    tensor_key = MODEL_TO_TENSOR_KEY[model_name]
    pipeline = pipelines[tensor_key]
    if model_name == "random_forest":
        rows, details, arrays = evaluate_random_forest(pipeline, rf_estimator)
    else:
        rows, details, arrays = evaluate_lightning_model(
            model_name=model_name,
            model=loaded_models[model_name],
            pipeline=pipeline,
        )
    for row in rows:
        row["tensor_key"] = tensor_key
    all_metrics_rows.extend(rows)
    prediction_details_by_model[model_name] = details
    prediction_delta_arrays_by_model[model_name] = arrays

evaluation_summary_df = pd.DataFrame(all_metrics_rows)
column_order = ["model_name", "tensor_key", "split", "n_samples"] + [
    column for column in evaluation_summary_df.columns
    if column not in {"model_name", "tensor_key", "split", "n_samples"}
]
evaluation_summary_df = evaluation_summary_df.loc[:, column_order]
display(evaluation_summary_df)
'''))

CELLS.append(md("""## Null Baselines on the L1000 and L1000_plate Test Sets

For reference we recompute the three null-baseline predictors (`zero`, `global_mean`,
`cell_line_mean`) against each pipeline's splits. This anchors the scale of every metric —
if a model does not beat `global_mean` or `cell_line_mean`, it has learned nothing useful
relative to an untrained average.
"""))

CELLS.append(code('''null_baseline_rows_by_tensor = {}
for tensor_key, pipeline in pipelines.items():
    data_module = pipeline["data_module"]
    baseline_rows = compute_null_baselines_from_loaders(
        train_loader=data_module.train_eval_dataloader(),
        val_loader=data_module.val_dataloader(),
        test_loader=data_module.test_dataloader(),
        gene_ids=pipeline["target_gene_ids"],
    )
    for row in baseline_rows:
        row["tensor_key"] = tensor_key
    null_baseline_rows_by_tensor[tensor_key] = baseline_rows

null_baseline_df = pd.DataFrame(
    [row for rows in null_baseline_rows_by_tensor.values() for row in rows]
)
null_baseline_df = null_baseline_df.loc[
    :,
    ["tensor_key", "predictor", "split", "n_samples",
     "delta_mse", "delta_mae", "treated_cosine",
     "delta_pearson_mean", "delta_spearman_mean",
     "top50_deg_match_count_mean", "signed_ndcg_at_50_mean",
     "mann_whitney_not_significant_fraction"],
]
display(null_baseline_df.sort_values(["tensor_key", "split", "predictor"], ignore_index=True))
'''))

CELLS.append(md("""## Retrieval Metric Comparison

`compute_condition_retrieval` measures whether each model's predicted Δ retrieves the correct
drug identity among all held-out drugs by cosine similarity against per-drug mean target Δ.
`top1` / `top5` should be compared against the printed `chance_top1` / `chance_top5`.
"""))

CELLS.append(code('''retrieval_rows = []
for model_name, arrays_by_split in prediction_delta_arrays_by_model.items():
    tensor_key = MODEL_TO_TENSOR_KEY[model_name]
    retrieval_label_column = resolve_retrieval_label_column(SPLIT_MODE)
    for split_name, (predicted_delta_np, target_delta_np, labels) in arrays_by_split.items():
        retrieval_row = {
            "model_name": model_name,
            "tensor_key": tensor_key,
            "split": split_name,
            "label_column": retrieval_label_column,
        }
        retrieval_row.update(
            compute_condition_retrieval(predicted_delta_np, target_delta_np, labels)
        )
        retrieval_rows.append(retrieval_row)
retrieval_summary_df = pd.DataFrame(retrieval_rows)
display(retrieval_summary_df)
'''))

CELLS.append(md("""## Head-to-Head Test Metric Table

A single table, one row per model, containing the headline test-set metrics. Use this as
the primary go/no-go comparison; per-row deltas from the best row are appended for each
metric where larger is better (so positive = worse than best).
"""))

CELLS.append(code('''TEST_HEADLINE_METRICS = [
    ("delta_pearson_mean", True),
    ("delta_spearman_mean", True),
    ("delta_cosine_mean", True),
    ("treated_cosine", True),
    ("top50_deg_match_count_mean", True),
    ("signed_ndcg_at_50_mean", True),
    ("delta_mse", False),
    ("delta_mae", False),
    ("mann_whitney_not_significant_fraction", True),
]

test_summary_df = evaluation_summary_df.loc[evaluation_summary_df["split"] == "test"].copy()
test_summary_df["model_order"] = test_summary_df["model_name"].map({name: idx for idx, name in enumerate(MODEL_DISPLAY_ORDER)})
test_summary_df = test_summary_df.sort_values("model_order", ignore_index=True).drop(columns=["model_order"])

test_headline_df = test_summary_df.loc[
    :,
    ["model_name", "tensor_key", "n_samples"] + [name for name, _ in TEST_HEADLINE_METRICS],
].copy()
for metric_name, higher_is_better in TEST_HEADLINE_METRICS:
    if higher_is_better:
        best_value = test_headline_df[metric_name].max()
        gap_column = f"{metric_name}_gap_to_best"
        test_headline_df[gap_column] = test_headline_df[metric_name] - best_value
    else:
        best_value = test_headline_df[metric_name].min()
        gap_column = f"{metric_name}_gap_to_best"
        test_headline_df[gap_column] = test_headline_df[metric_name] - best_value

display(test_headline_df)
'''))

CELLS.append(md("""## Headline Metric Bar Plots

The four most load-bearing metrics (delta Pearson, signed nDCG@50, top-50 DEG overlap,
treated cosine) on the test split, one model per bar. Dashed reference lines show the
strongest null baseline from `null_baseline_df` for the matching tensor set, so it is
immediately obvious whether each model beats "predict mean Δ per cell line".
"""))

CELLS.append(code('''HEADLINE_BAR_METRICS = [
    ("delta_pearson_mean", "Test Mean Δ Pearson"),
    ("signed_ndcg_at_50_mean", "Test Mean Signed nDCG@50"),
    ("top50_deg_match_count_mean", "Test Mean Top-50 DEG Overlap Count"),
    ("treated_cosine", "Test Treated-Expression Cosine"),
]

MODEL_COLOR_PALETTE = {
    "mlp": "#1f77b4",
    "random_forest": "#2ca02c",
    "adae_cell_line": "#9467bd",
    "adae_plate": "#d62728",
    "adae_plate_ablation_lambda0": "#ff7f0e",
}

null_baseline_test_df = null_baseline_df.loc[
    (null_baseline_df["split"] == "test") & (null_baseline_df["predictor"] == "cell_line_mean")
].set_index("tensor_key")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
plot_order = [name for name in MODEL_DISPLAY_ORDER if name in set(test_summary_df["model_name"])]
for ax, (metric_name, metric_title) in zip(axes, HEADLINE_BAR_METRICS):
    sns.barplot(
        data=test_summary_df,
        x="model_name",
        y=metric_name,
        order=plot_order,
        hue="model_name",
        palette=MODEL_COLOR_PALETTE,
        dodge=False,
        legend=False,
        ax=ax,
    )
    for tensor_key, baseline_row in null_baseline_test_df.iterrows():
        baseline_value = float(baseline_row[metric_name]) if metric_name in baseline_row else float("nan")
        if not np.isnan(baseline_value):
            ax.axhline(
                baseline_value,
                linestyle="--",
                linewidth=1.0,
                color="#555555" if tensor_key == L1000_TENSOR_KEY else "#aa5555",
                label=f"cell_line_mean ({tensor_key})",
            )
    ax.set_title(metric_title)
    ax.set_xlabel("")
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best", fontsize=8)
fig.suptitle(f"Test-Set Model Comparison ({SPLIT_MODE}, seed {RANDOM_SEED})", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
'''))

CELLS.append(md("""## Training Dynamics Overlay

Overlay the per-epoch `val_delta_pearson` (when logged) and `val_loss` curves from each
Lightning run, so it is easy to see where each model actually converged and where early
stopping kicked in. The random forest has no epoch-level trace (single-shot fit) and is
therefore omitted from this figure.
"""))

CELLS.append(code('''def _epoch_collapsed_series(metrics_df, column_name):
    if column_name not in metrics_df.columns:
        return None
    series_df = metrics_df.loc[metrics_df[column_name].notna(), ["epoch", column_name]].copy()
    if series_df.empty:
        return None
    series_df = series_df.sort_values("epoch")
    return series_df.groupby("epoch", as_index=False)[column_name].last()


training_curves_rows = []
for model_name in MODEL_DISPLAY_ORDER:
    if model_name not in MODEL_METRICS_CSV_PATHS:
        continue
    csv_path = resolve_project_path(MODEL_METRICS_CSV_PATHS[model_name])
    metrics_df = load_lightning_metrics_table(csv_path)
    for metric_column in ("val_loss", "val_delta_pearson", "val_treated_cosine", "train_loss"):
        collapsed = _epoch_collapsed_series(metrics_df, metric_column)
        if collapsed is None:
            continue
        for _, curve_row in collapsed.iterrows():
            training_curves_rows.append(
                {
                    "model_name": model_name,
                    "metric": metric_column,
                    "epoch": int(curve_row["epoch"]),
                    "value": float(curve_row[metric_column]),
                }
            )
training_curves_df = pd.DataFrame(training_curves_rows)

available_metrics = sorted(training_curves_df["metric"].unique().tolist())
preferred_panel_metrics = [
    metric for metric in ("val_delta_pearson", "val_loss", "val_treated_cosine", "train_loss")
    if metric in available_metrics
]
if not preferred_panel_metrics:
    raise ValueError("No recognizable validation / training metric columns were logged for any model.")

fig, axes = plt.subplots(1, len(preferred_panel_metrics), figsize=(6 * len(preferred_panel_metrics), 5), sharex=False)
if len(preferred_panel_metrics) == 1:
    axes = [axes]
for ax, metric_name in zip(axes, preferred_panel_metrics):
    metric_df = training_curves_df.loc[training_curves_df["metric"] == metric_name]
    sns.lineplot(
        data=metric_df,
        x="epoch",
        y="value",
        hue="model_name",
        hue_order=[name for name in MODEL_DISPLAY_ORDER if name in set(metric_df["model_name"])],
        palette=MODEL_COLOR_PALETTE,
        marker="o",
        ax=ax,
    )
    ax.set_title(f"{metric_name} by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
fig.suptitle(f"Training Dynamics ({SPLIT_MODE})", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
'''))

CELLS.append(md("""## Cell-Line Accuracy Under Mann-Whitney Non-Significance

Each prediction counts as "correct" if its per-row Mann-Whitney p-value is above 0.05 (i.e. the
predicted and true treated expression distributions are statistically indistinguishable). We
report per-cell-line accuracy across the test split for every model on the same axis so outliers
are obvious.
"""))

CELLS.append(code('''cell_line_accuracy_rows = []
for model_name in MODEL_DISPLAY_ORDER:
    details = prediction_details_by_model[model_name].get("test")
    if details is None or details.empty:
        continue
    cell_line_details = details.copy()
    cell_line_details["is_correct_prediction"] = cell_line_details["mann_whitney_pvalue"] > 0.05
    per_cell_line_accuracy = (
        cell_line_details.groupby("cell_line", as_index=False)
        .agg(
            n_test_samples=("condition_key", "size"),
            n_correct_predictions=("is_correct_prediction", "sum"),
        )
    )
    per_cell_line_accuracy["n_correct_predictions"] = per_cell_line_accuracy["n_correct_predictions"].astype(int)
    per_cell_line_accuracy["cell_line_accuracy"] = (
        per_cell_line_accuracy["n_correct_predictions"] / per_cell_line_accuracy["n_test_samples"]
    )
    per_cell_line_accuracy["model_name"] = model_name
    cell_line_accuracy_rows.append(per_cell_line_accuracy)
cell_line_accuracy_df = pd.concat(cell_line_accuracy_rows, ignore_index=True) if cell_line_accuracy_rows else pd.DataFrame()

mean_cell_line_accuracy_df = (
    cell_line_accuracy_df.groupby("model_name", as_index=False)["cell_line_accuracy"].mean()
    if not cell_line_accuracy_df.empty
    else pd.DataFrame()
)
display(mean_cell_line_accuracy_df)

if not cell_line_accuracy_df.empty:
    cell_line_order = (
        cell_line_accuracy_df.groupby("cell_line", as_index=False)["cell_line_accuracy"]
        .mean()
        .sort_values("cell_line_accuracy", ascending=False)["cell_line"]
        .tolist()
    )
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(
        data=cell_line_accuracy_df,
        x="cell_line",
        y="cell_line_accuracy",
        hue="model_name",
        hue_order=[name for name in MODEL_DISPLAY_ORDER if name in set(cell_line_accuracy_df["model_name"])],
        palette=MODEL_COLOR_PALETTE,
        order=cell_line_order,
        ax=ax,
    )
    ax.set_title("Test Cell-Line Accuracy (Mann-Whitney p > 0.05) by Model")
    ax.set_xlabel("Cell line")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    plt.show()
'''))

CELLS.append(md("""## Per-Plate Variance: Plate Adversary vs λ=0 Ablation

This is the deconfounding ask from notebook 8: does the plate adversary actually flatten the
per-plate dispersion of test-set metrics relative to the λ=0 ablation? We group each plate-tensor
model's per-row predictions by plate and compare the resulting per-plate std of `delta_pearson`
and `top50_deg_match_count`. Smaller std under the adversary would support the claim.
"""))

CELLS.append(code('''plate_pipeline = pipelines[L1000_PLATE_TENSOR_KEY]
plate_examples_df = plate_pipeline["test_examples_df"].loc[:, ["condition_key", "plate"]]

plate_comparison_rows = []
per_plate_frames = []
for model_name in ("adae_plate", "adae_plate_ablation_lambda0"):
    details = prediction_details_by_model[model_name].get("test")
    if details is None or details.empty:
        continue
    details_with_plate = details.merge(plate_examples_df, on="condition_key", how="left")
    if details_with_plate["plate"].isna().any():
        raise ValueError(f"Could not resolve plate for some rows in {model_name} test details.")

    per_plate_agg = (
        details_with_plate.groupby("plate", as_index=False)
        .agg(
            n_test_samples=("condition_key", "size"),
            plate_delta_pearson_mean=("delta_pearson", "mean"),
            plate_top50_deg_mean=("top50_deg_match_count", "mean"),
            plate_treated_cosine_mean=("treated_cosine", "mean"),
        )
    )
    per_plate_agg["model_name"] = model_name
    per_plate_frames.append(per_plate_agg)

    plate_comparison_rows.append(
        {
            "model_name": model_name,
            "n_plates": int(per_plate_agg.shape[0]),
            "delta_pearson_plate_std": float(per_plate_agg["plate_delta_pearson_mean"].std(ddof=0)),
            "delta_pearson_plate_range": float(
                per_plate_agg["plate_delta_pearson_mean"].max()
                - per_plate_agg["plate_delta_pearson_mean"].min()
            ),
            "top50_deg_plate_std": float(per_plate_agg["plate_top50_deg_mean"].std(ddof=0)),
            "treated_cosine_plate_std": float(per_plate_agg["plate_treated_cosine_mean"].std(ddof=0)),
        }
    )

plate_comparison_df = pd.DataFrame(plate_comparison_rows)
display(plate_comparison_df)

if per_plate_frames:
    per_plate_df = pd.concat(per_plate_frames, ignore_index=True)
    plate_order = (
        per_plate_df.groupby("plate", as_index=False)["plate_delta_pearson_mean"]
        .mean()
        .sort_values("plate_delta_pearson_mean", ascending=False)["plate"]
        .tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric_column, metric_title in (
        (axes[0], "plate_delta_pearson_mean", "Per-Plate Mean Δ Pearson (Test)"),
        (axes[1], "plate_top50_deg_mean", "Per-Plate Mean Top-50 DEG Overlap (Test)"),
    ):
        sns.barplot(
            data=per_plate_df,
            x="plate",
            y=metric_column,
            hue="model_name",
            hue_order=["adae_plate", "adae_plate_ablation_lambda0"],
            palette=MODEL_COLOR_PALETTE,
            order=plate_order,
            ax=ax,
        )
        ax.set_title(metric_title)
        ax.set_xlabel("Plate")
        ax.set_ylabel(metric_column)
        ax.tick_params(axis="x", rotation=60)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Plate Adversary vs λ=0 Ablation: Per-Plate Test Metrics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
'''))

CELLS.append(md("""## Linear Plate Probe on z_fused

For the plate-adversary and ablation models we run `probe_latent_plate_accuracy` — a logistic
regression fit to the frozen `z_fused` embeddings predicting the plate id. The adversary should
drive probe accuracy toward the random / majority baseline; the ablation should leave plate
identity linearly decodable.
"""))

CELLS.append(code('''plate_data_module = plate_pipeline["data_module"]

probe_rows = []
for model_name in ("adae_plate", "adae_plate_ablation_lambda0"):
    module = loaded_models[model_name]
    probe_result = probe_latent_plate_accuracy(
        module=module,
        fit_dataloader=plate_data_module.train_eval_dataloader(),
        eval_dataloader=plate_data_module.test_dataloader(),
        confounder_key="confounder_index",
    )
    probe_result["model_name"] = model_name
    probe_rows.append(probe_result)

probe_df = pd.DataFrame(probe_rows)
probe_df = probe_df.loc[
    :,
    ["model_name", "probe_accuracy", "random_baseline", "majority_baseline",
     "n_train_samples", "n_eval_samples", "n_train_classes", "n_eval_classes", "latent_dim"],
]
display(probe_df)

if not probe_df.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=probe_df,
        x="model_name",
        y="probe_accuracy",
        hue="model_name",
        palette=MODEL_COLOR_PALETTE,
        dodge=False,
        legend=False,
        ax=ax,
    )
    random_baseline = float(probe_df["random_baseline"].iloc[0])
    majority_baseline = float(probe_df["majority_baseline"].max())
    ax.axhline(random_baseline, linestyle="--", color="#555555", linewidth=1.0,
               label=f"random = {random_baseline:.3f}")
    ax.axhline(majority_baseline, linestyle="--", color="#d62728", linewidth=1.0,
               label=f"majority = {majority_baseline:.3f}")
    ax.set_ylim(0, max(1.0, float(probe_df["probe_accuracy"].max()) * 1.1))
    ax.set_title("Linear Plate Probe Accuracy on z_fused (Test)")
    ax.set_xlabel("")
    ax.set_ylabel("Probe accuracy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    plt.show()
'''))

CELLS.append(md("""## Summary Takeaways

The tables and plots above are the comparison; this cell consolidates the one-line verdict on each
model so the reader does not have to rederive it from the summary frames:
"""))

CELLS.append(code('''def _lookup_metric(test_df, model_name, column):
    row = test_df.loc[test_df["model_name"] == model_name]
    if row.empty or column not in row.columns:
        return float("nan")
    return float(row.iloc[0][column])


verdict_rows = []
for model_name in MODEL_DISPLAY_ORDER:
    verdict_rows.append(
        {
            "model_name": model_name,
            "tensor_key": MODEL_TO_TENSOR_KEY[model_name],
            "test_n_samples": int(_lookup_metric(test_summary_df, model_name, "n_samples"))
                if model_name in set(test_summary_df["model_name"]) else 0,
            "test_delta_pearson_mean": _lookup_metric(test_summary_df, model_name, "delta_pearson_mean"),
            "test_signed_ndcg_at_50_mean": _lookup_metric(test_summary_df, model_name, "signed_ndcg_at_50_mean"),
            "test_top50_deg_match_count_mean": _lookup_metric(test_summary_df, model_name, "top50_deg_match_count_mean"),
            "test_treated_cosine": _lookup_metric(test_summary_df, model_name, "treated_cosine"),
            "test_mann_whitney_not_sig_fraction": _lookup_metric(
                test_summary_df, model_name, "mann_whitney_not_significant_fraction"
            ),
        }
    )
verdict_df = pd.DataFrame(verdict_rows)
display(verdict_df)

print(
    "Reminder: `adae_plate` / `adae_plate_ablation_lambda0` evaluate against the L1000_plate "
    "test split (different condition rows than the L1000 split used by `mlp` / `random_forest` / "
    "`adae_cell_line`). Use the tensor_key column above to interpret cross-tensor deltas."
)
'''))


def main():
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path = Path(__file__).resolve().parent / "9_model_comparison.ipynb"
    output_path.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {output_path} with {len(CELLS)} cells.")


if __name__ == "__main__":
    main()
