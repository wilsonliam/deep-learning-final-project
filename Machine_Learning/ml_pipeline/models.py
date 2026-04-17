import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


class RidgeDrugResponseModule(L.LightningModule):
    def __init__(
        self,
        input_dim,
        gene_dim,
        learning_rate,
        weight_decay,
        delta_mean,
        delta_std,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "input_dim": int(input_dim),
                "gene_dim": int(gene_dim),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
            }
        )
        self.input_dim = int(input_dim)
        self.gene_dim = int(gene_dim)
        self.output_dim = int(gene_dim)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.register_buffer("delta_mean", delta_mean.to(torch.float32))
        self.register_buffer("delta_std", delta_std.to(torch.float32))
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, input_features):
        standardized_delta = self.linear(input_features)
        return standardized_delta * self.delta_std + self.delta_mean

    def _shared_step(self, batch, stage):
        predicted_delta = self(batch["input_features"])
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


MODEL_REGISTRY = {
    "ridge": RidgeDrugResponseModule,
}


def build_model(model_name, **model_kwargs):
    normalized_model_name = str(model_name).strip().lower()
    if normalized_model_name not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}. Available models: {available_models}")
    return MODEL_REGISTRY[normalized_model_name](**model_kwargs)
