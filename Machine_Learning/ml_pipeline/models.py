import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


def _as_tuple(hidden_dims):
    return tuple(int(hidden_dim) for hidden_dim in hidden_dims)


def _build_mlp(input_dim, hidden_dims, output_dim, activation_cls=nn.ReLU):
    layers = []
    previous_dim = int(input_dim)

    for hidden_dim in _as_tuple(hidden_dims):
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(activation_cls())
        previous_dim = hidden_dim

    layers.append(nn.Linear(previous_dim, int(output_dim)))
    return nn.Sequential(*layers)


class BaseDrugResponseModule(L.LightningModule):
    def __init__(self, gene_dim, learning_rate, weight_decay, delta_mean, delta_std, **saved_hparams):
        super().__init__()
        serialized_hparams = {
            key: tuple(value) if isinstance(value, (list, tuple)) else value
            for key, value in saved_hparams.items()
        }
        serialized_hparams.update(
            {
                "gene_dim": int(gene_dim),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
            }
        )
        self.save_hyperparameters(serialized_hparams)
        self.gene_dim = int(gene_dim)
        self.output_dim = int(gene_dim)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.model_family = str(serialized_hparams.get("model_family", "unknown"))
        self.register_buffer("delta_mean", delta_mean.to(torch.float32))
        self.register_buffer("delta_std", delta_std.to(torch.float32))

    def predict_delta(self, model_inputs):
        raise NotImplementedError

    def forward(self, model_inputs):
        return self.predict_delta(model_inputs)

    def _shared_step(self, batch, stage):
        predicted_delta = self.predict_delta(batch)
        target_delta = batch["target_delta"]
        predicted_expression = batch["baseline_expression"] + predicted_delta
        target_expression = batch["baseline_expression"] + target_delta
        loss = F.mse_loss(predicted_delta, target_delta)
        delta_mae = F.l1_loss(predicted_delta, target_delta)
        treated_cosine = F.cosine_similarity(predicted_expression, target_expression, dim=1).mean()

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=stage != "train", batch_size=target_delta.shape[0])
        self.log(f"{stage}_delta_mse", loss, on_step=False, on_epoch=True, batch_size=target_delta.shape[0])
        self.log(f"{stage}_delta_mae", delta_mae, on_step=False, on_epoch=True, batch_size=target_delta.shape[0])
        self.log(f"{stage}_treated_cosine", treated_cosine, on_step=False, on_epoch=True, prog_bar=stage != "train", batch_size=target_delta.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith("bias"):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )
        return optimizer


class RidgeDrugResponseModule(BaseDrugResponseModule):
    def __init__(
        self,
        input_dim,
        gene_dim,
        learning_rate,
        weight_decay,
        delta_mean,
        delta_std,
    ):
        self.input_dim = int(input_dim)
        super().__init__(
            gene_dim=gene_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            delta_mean=delta_mean,
            delta_std=delta_std,
            model_family="ridge",
            input_dim=self.input_dim,
        )
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def predict_delta(self, model_inputs):
        if isinstance(model_inputs, dict):
            input_features = model_inputs["input_features"]
        else:
            input_features = model_inputs

        standardized_delta = self.linear(input_features)
        return standardized_delta * self.delta_std + self.delta_mean


class MLPDrugResponseModule(BaseDrugResponseModule):
    def __init__(
        self,
        input_gene_dim,
        drug_feature_dim,
        dose_feature_dim,
        gene_branch_hidden_dims,
        drug_branch_hidden_dims,
        combined_hidden_dims,
        gene_dim,
        learning_rate,
        weight_decay,
        delta_mean,
        delta_std,
    ):
        self.input_gene_dim = int(input_gene_dim)
        self.drug_feature_dim = int(drug_feature_dim)
        self.dose_feature_dim = int(dose_feature_dim)
        self.gene_branch_hidden_dims = _as_tuple(gene_branch_hidden_dims)
        self.drug_branch_hidden_dims = _as_tuple(drug_branch_hidden_dims)
        self.combined_hidden_dims = _as_tuple(combined_hidden_dims)

        if not self.gene_branch_hidden_dims:
            raise ValueError("gene_branch_hidden_dims must contain at least one hidden layer.")
        if not self.drug_branch_hidden_dims:
            raise ValueError("drug_branch_hidden_dims must contain at least one hidden layer.")
        if not self.combined_hidden_dims:
            raise ValueError("combined_hidden_dims must contain at least one hidden layer.")

        super().__init__(
            gene_dim=gene_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            delta_mean=delta_mean,
            delta_std=delta_std,
            model_family="mlp",
            input_gene_dim=self.input_gene_dim,
            drug_feature_dim=self.drug_feature_dim,
            dose_feature_dim=self.dose_feature_dim,
            gene_branch_hidden_dims=self.gene_branch_hidden_dims,
            drug_branch_hidden_dims=self.drug_branch_hidden_dims,
            combined_hidden_dims=self.combined_hidden_dims,
        )

        self.gene_network = _build_mlp(
            input_dim=self.input_gene_dim,
            hidden_dims=self.gene_branch_hidden_dims[:-1],
            output_dim=self.gene_branch_hidden_dims[-1],
        )
        self.drug_network = _build_mlp(
            input_dim=self.drug_feature_dim,
            hidden_dims=self.drug_branch_hidden_dims[:-1],
            output_dim=self.drug_branch_hidden_dims[-1],
        )
        combined_input_dim = self.gene_branch_hidden_dims[-1] + self.drug_branch_hidden_dims[-1] + self.dose_feature_dim
        self.combined_network = _build_mlp(
            input_dim=combined_input_dim,
            hidden_dims=self.combined_hidden_dims,
            output_dim=self.output_dim,
        )

    def _split_input_features(self, input_features):
        gene_end = self.input_gene_dim
        drug_end = gene_end + self.drug_feature_dim
        gene_features = input_features[:, :gene_end]
        drug_features = input_features[:, gene_end:drug_end]
        dose_feature = input_features[:, drug_end:]
        if dose_feature.ndim == 1:
            dose_feature = dose_feature.unsqueeze(1)
        return gene_features, drug_features, dose_feature

    def _resolve_branch_inputs(self, model_inputs):
        if isinstance(model_inputs, dict):
            gene_features = model_inputs["gene_features"]
            drug_features = model_inputs["drug_features"]
            dose_feature = model_inputs["dose_feature"]
        else:
            gene_features, drug_features, dose_feature = self._split_input_features(model_inputs)

        if dose_feature.ndim == 1:
            dose_feature = dose_feature.unsqueeze(1)

        return gene_features, drug_features, dose_feature

    def predict_delta(self, model_inputs):
        gene_features, drug_features, dose_feature = self._resolve_branch_inputs(model_inputs)
        gene_embedding = self.gene_network(gene_features)
        drug_embedding = self.drug_network(drug_features)
        combined_features = torch.cat([gene_embedding, drug_embedding, dose_feature], dim=1)
        standardized_delta = self.combined_network(combined_features)
        return standardized_delta * self.delta_std + self.delta_mean


MODEL_REGISTRY = {
    "ridge": RidgeDrugResponseModule,
    "mlp": MLPDrugResponseModule,
}


def build_model(model_name, **model_kwargs):
    normalized_model_name = str(model_name).strip().lower()
    if normalized_model_name not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}. Available models: {available_models}")
    return MODEL_REGISTRY[normalized_model_name](**model_kwargs)
