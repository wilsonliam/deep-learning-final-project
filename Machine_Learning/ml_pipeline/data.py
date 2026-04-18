import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import MIN_STANDARD_DEVIATION, build_row_slices


class TreatmentExampleDataset(Dataset):
    def __init__(self, examples_df, dmso_bundle, treatment_bundle, fingerprint_bundle):
        self.examples_df = examples_df.reset_index(drop=True).copy()
        self.dmso_bundle = dmso_bundle
        self.treatment_bundle = treatment_bundle
        self.fingerprint_bundle = fingerprint_bundle

    def __len__(self):
        return len(self.examples_df)

    def __getitem__(self, idx):
        row = self.examples_df.iloc[idx]
        baseline_index = int(row["baseline_index"])
        fingerprint_index = int(row["fingerprint_index"])
        target_index = int(row["target_index"])

        return {
            "baseline_expression": self.dmso_bundle["expressions"][baseline_index],
            "drug_fingerprint": self.fingerprint_bundle["fingerprints"][fingerprint_index],
            "concentration": torch.tensor(float(row["concentration"]), dtype=torch.float32),
            "target_expression": self.treatment_bundle["expressions"][target_index],
            "condition_key": row["condition_key"],
            "cell_line": row["cell_line"],
            "cell_name": row["cell_name"],
            "organ": row["organ"],
            "drug": row["drug"],
            "concentration_unit": row["concentration_unit"],
        }


class TrainingPreprocessor:
    def __init__(
        self,
        train_examples_df,
        dmso_bundle,
        treatment_bundle,
        fingerprint_bundle,
        target_gene_indices,
        target_mode,
        batch_size,
    ):
        self.train_examples_df = train_examples_df.reset_index(drop=True).copy()
        self.dmso_input_expression_lookup = dmso_bundle["expressions"].detach().cpu().to(torch.float32).contiguous()
        self.treatment_expression_lookup = treatment_bundle["expressions"].detach().cpu().to(torch.float32).contiguous()
        self.fingerprint_lookup = fingerprint_bundle["fingerprints"].detach().cpu().to(torch.float32).contiguous()
        self.target_gene_indices = torch.as_tensor(np.asarray(target_gene_indices, dtype=np.int64), dtype=torch.long)
        self.dmso_target_expression_lookup = self.dmso_input_expression_lookup.index_select(1, self.target_gene_indices).contiguous()
        self.target_mode = str(target_mode)
        self.batch_size = int(batch_size)
        self.input_gene_dim = int(self.dmso_input_expression_lookup.shape[1])
        self.target_gene_dim = int(self.treatment_expression_lookup.shape[1])
        self.gene_dim = int(self.target_gene_dim)
        self.fingerprint_dim = int(self.fingerprint_lookup.shape[1])
        self.input_dim = int(self.input_gene_dim + self.fingerprint_dim + 1)
        self.delta_mean = None
        self.delta_std = None
        self.baseline_mean = None
        self.baseline_std = None
        self.dose_log_mean = None
        self.dose_log_std = None
        self.normalized_dmso_input_expression_lookup = None

        if self.dmso_target_expression_lookup.shape[1] != self.target_gene_dim:
            raise ValueError("Resolved target gene indices do not match the treatment target width.")

    def _iter_train_batches(self):
        for start, end in build_row_slices(
            len(self.train_examples_df),
            batch_size=self.batch_size,
            min_last_batch_size=1,
        ):
            yield self.train_examples_df.iloc[start:end]

    def _get_input_baseline_batch(self, batch_df):
        baseline_indices = torch.as_tensor(batch_df["baseline_index"].to_numpy(np.int64), dtype=torch.long)
        return self.dmso_input_expression_lookup[baseline_indices]

    def _get_target_baseline_batch(self, batch_df):
        baseline_indices = torch.as_tensor(batch_df["baseline_index"].to_numpy(np.int64), dtype=torch.long)
        return self.dmso_target_expression_lookup[baseline_indices]

    def _get_delta_batch(self, batch_df):
        baseline_batch = self._get_target_baseline_batch(batch_df)
        target_indices = torch.as_tensor(batch_df["target_index"].to_numpy(np.int64), dtype=torch.long)
        target_batch = self.treatment_expression_lookup[target_indices]
        return target_batch - baseline_batch

    def fit(self):
        train_count = len(self.train_examples_df)
        if train_count <= 1:
            raise ValueError("Need at least two training examples to fit the preprocessor.")

        baseline_sum = torch.zeros(self.input_gene_dim, dtype=torch.float64)
        baseline_sq_sum = torch.zeros(self.input_gene_dim, dtype=torch.float64)
        for batch_df in self._iter_train_batches():
            baseline_batch = self._get_input_baseline_batch(batch_df).to(torch.float64)
            baseline_sum += baseline_batch.sum(dim=0)
            baseline_sq_sum += baseline_batch.square().sum(dim=0)

        baseline_mean = baseline_sum / train_count
        baseline_var = baseline_sq_sum / train_count - baseline_mean.square()
        baseline_std = torch.sqrt(torch.clamp(baseline_var, min=MIN_STANDARD_DEVIATION))
        self.baseline_mean = baseline_mean.to(torch.float32)
        self.baseline_std = baseline_std.to(torch.float32)
        self.normalized_dmso_input_expression_lookup = (
            (self.dmso_input_expression_lookup - self.baseline_mean) / self.baseline_std
        ).to(torch.float32)

        dose_values = self.train_examples_df["concentration"].to_numpy(np.float32)
        dose_log_values = np.log10(np.clip(dose_values, a_min=MIN_STANDARD_DEVIATION, a_max=None))
        self.dose_log_mean = torch.tensor(float(dose_log_values.mean()), dtype=torch.float32)
        self.dose_log_std = torch.tensor(
            float(max(dose_log_values.std(), MIN_STANDARD_DEVIATION)),
            dtype=torch.float32,
        )

        delta_sum = torch.zeros(self.target_gene_dim, dtype=torch.float64)
        delta_sq_sum = torch.zeros(self.target_gene_dim, dtype=torch.float64)
        for batch_df in self._iter_train_batches():
            delta_batch = self._get_delta_batch(batch_df).to(torch.float64)
            delta_sum += delta_batch.sum(dim=0)
            delta_sq_sum += delta_batch.square().sum(dim=0)

        delta_mean = delta_sum / train_count
        delta_var = delta_sq_sum / train_count - delta_mean.square()
        delta_std = torch.sqrt(torch.clamp(delta_var, min=MIN_STANDARD_DEVIATION))
        self.delta_mean = delta_mean.to(torch.float32)
        self.delta_std = delta_std.to(torch.float32)
        return self

    def summary_frame(self):
        return pd.DataFrame(
            [
                {
                    "target_mode": self.target_mode,
                    "train_examples": int(len(self.train_examples_df)),
                    "input_dim": int(self.input_dim),
                    "input_gene_dim": int(self.input_gene_dim),
                    "target_gene_dim": int(self.target_gene_dim),
                    "fingerprint_dim": int(self.fingerprint_dim),
                    "dose_log_mean": float(self.dose_log_mean.item()),
                    "dose_log_std": float(self.dose_log_std.item()),
                    "baseline_std_min": float(self.baseline_std.min().item()),
                    "delta_std_min": float(self.delta_std.min().item()),
                }
            ]
        )


class PreparedTreatmentDataset(Dataset):
    def __init__(
        self,
        examples_df,
        preprocessor,
        cell_line_to_index=None,
        confounder_column=None,
        confounder_to_index=None,
    ):
        self.examples_df = examples_df.reset_index(drop=True).copy()
        self.preprocessor = preprocessor
        self.condition_keys = self.examples_df["condition_key"].tolist()
        self.cell_lines = self.examples_df["cell_line"].tolist()
        self.cell_names = self.examples_df["cell_name"].tolist()
        self.organs = self.examples_df["organ"].tolist()
        self.drugs = self.examples_df["drug"].tolist()
        self.concentration_units = self.examples_df["concentration_unit"].tolist()
        self.concentrations = torch.tensor(
            self.examples_df["concentration"].to_numpy(np.float32),
            dtype=torch.float32,
        )
        self.scaled_doses = (
            (torch.log10(torch.clamp(self.concentrations, min=MIN_STANDARD_DEVIATION)) - self.preprocessor.dose_log_mean)
            / self.preprocessor.dose_log_std
        ).to(torch.float32)
        self.baseline_indices = torch.tensor(
            self.examples_df["baseline_index"].to_numpy(np.int64),
            dtype=torch.long,
        )
        self.fingerprint_indices = torch.tensor(
            self.examples_df["fingerprint_index"].to_numpy(np.int64),
            dtype=torch.long,
        )
        self.target_indices = torch.tensor(
            self.examples_df["target_index"].to_numpy(np.int64),
            dtype=torch.long,
        )

        self.cell_line_to_index = dict(cell_line_to_index) if cell_line_to_index is not None else None
        if self.cell_line_to_index is not None:
            unmapped_cell_lines = sorted({cl for cl in self.cell_lines if cl not in self.cell_line_to_index})
            if unmapped_cell_lines:
                raise KeyError(
                    f"cell_line_to_index is missing entries for {len(unmapped_cell_lines)} cell line(s): "
                    f"{unmapped_cell_lines[:5]}"
                )
            self.cell_line_indices = torch.tensor(
                [int(self.cell_line_to_index[cl]) for cl in self.cell_lines],
                dtype=torch.long,
            )
        else:
            self.cell_line_indices = None

        if (confounder_column is None) != (confounder_to_index is None):
            raise ValueError(
                "confounder_column and confounder_to_index must be provided together."
            )
        if confounder_column is None and confounder_to_index is None and self.cell_line_to_index is not None:
            confounder_column = "cell_line"
            confounder_to_index = self.cell_line_to_index

        self.confounder_column = str(confounder_column) if confounder_column is not None else None
        self.confounder_to_index = dict(confounder_to_index) if confounder_to_index is not None else None
        if self.confounder_to_index is not None:
            if self.confounder_column not in self.examples_df.columns:
                raise KeyError(
                    f"examples_df is missing the confounder column '{self.confounder_column}'."
                )
            confounder_values = self.examples_df[self.confounder_column].tolist()
            unmapped_confounders = sorted({v for v in confounder_values if v not in self.confounder_to_index})
            if unmapped_confounders:
                raise KeyError(
                    f"confounder_to_index is missing entries for {len(unmapped_confounders)} "
                    f"'{self.confounder_column}' value(s): {unmapped_confounders[:5]}"
                )
            self.confounder_values = confounder_values
            self.confounder_indices = torch.tensor(
                [int(self.confounder_to_index[v]) for v in confounder_values],
                dtype=torch.long,
            )
        else:
            self.confounder_values = None
            self.confounder_indices = None

    def __len__(self):
        return len(self.examples_df)

    def __getitem__(self, idx):
        baseline_index = int(self.baseline_indices[idx])
        fingerprint_index = int(self.fingerprint_indices[idx])
        target_index = int(self.target_indices[idx])

        baseline_expression = self.preprocessor.dmso_target_expression_lookup[baseline_index]
        target_expression = self.preprocessor.treatment_expression_lookup[target_index]
        target_delta = target_expression - baseline_expression
        gene_features = self.preprocessor.normalized_dmso_input_expression_lookup[baseline_index]
        drug_features = self.preprocessor.fingerprint_lookup[fingerprint_index]
        dose_feature = self.scaled_doses[idx].view(1)
        input_features = torch.cat(
            [gene_features, drug_features, dose_feature],
            dim=0,
        )

        item = {
            "dataset_index": int(idx),
            "input_features": input_features,
            "gene_features": gene_features,
            "drug_features": drug_features,
            "dose_feature": dose_feature,
            "baseline_expression": baseline_expression,
            "target_delta": target_delta,
            "concentration": self.concentrations[idx],
            "condition_key": self.condition_keys[idx],
            "cell_line": self.cell_lines[idx],
            "cell_name": self.cell_names[idx],
            "organ": self.organs[idx],
            "drug": self.drugs[idx],
            "concentration_unit": self.concentration_units[idx],
        }
        if self.cell_line_indices is not None:
            item["cell_line_index"] = self.cell_line_indices[idx]
        if self.confounder_indices is not None:
            item["confounder_index"] = self.confounder_indices[idx]
        return item


class DrugResponseDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_examples_df,
        val_examples_df,
        test_examples_df,
        preprocessor,
        batch_size,
        num_workers,
        seed,
        cell_line_to_index=None,
        confounder_column=None,
        confounder_to_index=None,
    ):
        super().__init__()
        self.train_examples_df = train_examples_df.reset_index(drop=True).copy()
        self.val_examples_df = val_examples_df.reset_index(drop=True).copy()
        self.test_examples_df = test_examples_df.reset_index(drop=True).copy()
        self.preprocessor = preprocessor
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.cell_line_to_index = dict(cell_line_to_index) if cell_line_to_index is not None else None
        if (confounder_column is None) != (confounder_to_index is None):
            raise ValueError(
                "confounder_column and confounder_to_index must be provided together."
            )
        self.confounder_column = str(confounder_column) if confounder_column is not None else None
        self.confounder_to_index = dict(confounder_to_index) if confounder_to_index is not None else None
        self.pin_memory = bool(torch.cuda.is_available())
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _make_prepared_dataset(self, examples_df):
        return PreparedTreatmentDataset(
            examples_df,
            self.preprocessor,
            cell_line_to_index=self.cell_line_to_index,
            confounder_column=self.confounder_column,
            confounder_to_index=self.confounder_to_index,
        )

    def setup(self, stage=None):
        if self.train_dataset is None:
            self.train_dataset = self._make_prepared_dataset(self.train_examples_df)
        if self.val_dataset is None:
            self.val_dataset = self._make_prepared_dataset(self.val_examples_df)
        if self.test_dataset is None:
            self.test_dataset = self._make_prepared_dataset(self.test_examples_df)

    def _make_loader(self, dataset, shuffle):
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0,
        }
        if shuffle:
            loader_kwargs["generator"] = torch.Generator().manual_seed(self.seed)
        return DataLoader(**loader_kwargs)

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def train_eval_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=False)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)


RidgeResponseDataModule = DrugResponseDataModule
