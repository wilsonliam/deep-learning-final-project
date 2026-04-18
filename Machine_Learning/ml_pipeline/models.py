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


def _last_linear(module):
    for layer in reversed(tuple(module.modules())):
        if isinstance(layer, nn.Linear):
            return layer
    raise ValueError(f"Module {module.__class__.__name__} does not contain a Linear layer.")


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        return _GradientReversalFn.apply(x, self.alpha)


class FiLMFuser(nn.Module):
    def __init__(self, z_ctx_dim, z_drug_dim, fused_hidden_dims, activation_cls=nn.ReLU):
        super().__init__()
        self.z_ctx_dim = int(z_ctx_dim)
        self.z_drug_dim = int(z_drug_dim)
        self.fused_hidden_dims = _as_tuple(fused_hidden_dims)

        if not self.fused_hidden_dims:
            raise ValueError("fused_hidden_dims must contain at least one hidden layer.")

        conditioner_hidden_dims = self.fused_hidden_dims[:-1]
        self.conditioner = _build_mlp(
            input_dim=self.z_drug_dim,
            hidden_dims=conditioner_hidden_dims,
            output_dim=2 * self.z_ctx_dim,
            activation_cls=activation_cls,
        )
        conditioner_output = _last_linear(self.conditioner)
        # Keep FiLM close to identity at init, but not exactly identity, so z_drug
        # still receives a small gradient through the fused path on the first step.
        nn.init.normal_(conditioner_output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(conditioner_output.bias)
        with torch.no_grad():
            conditioner_output.bias[: self.z_ctx_dim].fill_(1.0)

        if len(self.fused_hidden_dims) == 1 and self.z_ctx_dim == self.fused_hidden_dims[-1]:
            self.post_fusion = nn.Identity()
        else:
            self.post_fusion = _build_mlp(
                input_dim=self.z_ctx_dim,
                hidden_dims=self.fused_hidden_dims[:-1],
                output_dim=self.fused_hidden_dims[-1],
                activation_cls=activation_cls,
            )

    def forward(self, z_ctx, z_drug):
        gamma_beta = self.conditioner(z_drug)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        modulated_ctx = gamma * z_ctx + beta
        return self.post_fusion(modulated_ctx)


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


class ADAEDrugResponseModule(BaseDrugResponseModule):
    def __init__(
        self,
        input_gene_dim,
        drug_feature_dim,
        dose_feature_dim,
        ctx_branch_hidden_dims,
        drug_branch_hidden_dims,
        z_ctx_dim,
        z_drug_dim,
        fused_hidden_dims,
        predictor_hidden_dims,
        adv_hidden_dims,
        n_confounder_classes,
        alpha_target,
        ema_momentum,
        lambda_min,
        lambda_max,
        gene_dim,
        learning_rate,
        weight_decay,
        delta_mean,
        delta_std,
        fuser_type="concat",
        adversary_conditioning="none",
    ):
        self.input_gene_dim = int(input_gene_dim)
        self.drug_feature_dim = int(drug_feature_dim)
        self.dose_feature_dim = int(dose_feature_dim)
        self.ctx_branch_hidden_dims = _as_tuple(ctx_branch_hidden_dims)
        self.drug_branch_hidden_dims = _as_tuple(drug_branch_hidden_dims)
        self.fused_hidden_dims = _as_tuple(fused_hidden_dims)
        self.predictor_hidden_dims = _as_tuple(predictor_hidden_dims)
        self.adv_hidden_dims = _as_tuple(adv_hidden_dims)
        self.z_ctx_dim = int(z_ctx_dim)
        self.z_drug_dim = int(z_drug_dim)
        self.n_confounder_classes = int(n_confounder_classes)
        self.alpha_target = float(alpha_target)
        self.ema_momentum = float(ema_momentum)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.fuser_type = str(fuser_type).strip().lower()
        self.adversary_conditioning = str(adversary_conditioning).strip().lower()
        self.ema_epsilon = 1e-8

        if not self.ctx_branch_hidden_dims:
            raise ValueError("ctx_branch_hidden_dims must contain at least one hidden layer.")
        if not self.drug_branch_hidden_dims:
            raise ValueError("drug_branch_hidden_dims must contain at least one hidden layer.")
        if not self.fused_hidden_dims:
            raise ValueError("fused_hidden_dims must contain at least one hidden layer.")
        if not self.predictor_hidden_dims:
            raise ValueError("predictor_hidden_dims must contain at least one hidden layer.")
        if not self.adv_hidden_dims:
            raise ValueError("adv_hidden_dims must contain at least one hidden layer.")
        if self.n_confounder_classes < 2:
            raise ValueError("n_confounder_classes must be at least 2 for a softmax+CE adversary.")
        if not (self.lambda_min >= 0 and self.lambda_max >= self.lambda_min):
            raise ValueError(
                "Require 0 <= lambda_min <= lambda_max. "
                "Set lambda_min == lambda_max to pin lambda to a constant (e.g. 0.0 for the adversary-off ablation)."
            )
        if not (0.0 <= self.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0, 1).")
        if self.fuser_type not in {"concat", "film"}:
            raise ValueError("fuser_type must be one of {'concat', 'film'}.")
        if self.adversary_conditioning not in {"none", "drug"}:
            raise ValueError("adversary_conditioning must be one of {'none', 'drug'}.")

        super().__init__(
            gene_dim=gene_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            delta_mean=delta_mean,
            delta_std=delta_std,
            model_family="adae",
            input_gene_dim=self.input_gene_dim,
            drug_feature_dim=self.drug_feature_dim,
            dose_feature_dim=self.dose_feature_dim,
            ctx_branch_hidden_dims=self.ctx_branch_hidden_dims,
            drug_branch_hidden_dims=self.drug_branch_hidden_dims,
            z_ctx_dim=self.z_ctx_dim,
            z_drug_dim=self.z_drug_dim,
            fused_hidden_dims=self.fused_hidden_dims,
            predictor_hidden_dims=self.predictor_hidden_dims,
            adv_hidden_dims=self.adv_hidden_dims,
            n_confounder_classes=self.n_confounder_classes,
            alpha_target=self.alpha_target,
            ema_momentum=self.ema_momentum,
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max,
            fuser_type=self.fuser_type,
            adversary_conditioning=self.adversary_conditioning,
        )

        self.encoder_ctx = _build_mlp(
            input_dim=self.input_gene_dim + self.dose_feature_dim,
            hidden_dims=self.ctx_branch_hidden_dims,
            output_dim=self.z_ctx_dim,
        )
        self.encoder_drug = _build_mlp(
            input_dim=self.drug_feature_dim + self.dose_feature_dim,
            hidden_dims=self.drug_branch_hidden_dims,
            output_dim=self.z_drug_dim,
        )
        self.fused_output_dim = int(self.fused_hidden_dims[-1])
        if self.fuser_type == "concat":
            self.fuser = _build_mlp(
                input_dim=self.z_ctx_dim + self.z_drug_dim,
                hidden_dims=self.fused_hidden_dims[:-1],
                output_dim=self.fused_output_dim,
            )
        else:
            self.fuser = FiLMFuser(
                z_ctx_dim=self.z_ctx_dim,
                z_drug_dim=self.z_drug_dim,
                fused_hidden_dims=self.fused_hidden_dims,
            )
        self.predictor_head = _build_mlp(
            input_dim=self.fused_output_dim,
            hidden_dims=self.predictor_hidden_dims,
            output_dim=self.output_dim,
        )
        self.grl = GradientReversalLayer(alpha=self.alpha_target)
        adversary_input_dim = self.fused_output_dim
        if self.adversary_conditioning == "drug":
            adversary_input_dim += self.z_drug_dim
        self.adversary_head = _build_mlp(
            input_dim=adversary_input_dim,
            hidden_dims=self.adv_hidden_dims,
            output_dim=self.n_confounder_classes,
        )

        self.register_buffer("ema_mse_std", torch.zeros(()))
        self.register_buffer("ema_adv_ce", torch.zeros(()))
        self.register_buffer("ema_initialized", torch.zeros((), dtype=torch.bool))

    def _resolve_branch_inputs(self, model_inputs):
        if isinstance(model_inputs, dict):
            gene_features = model_inputs["gene_features"]
            drug_features = model_inputs["drug_features"]
            dose_feature = model_inputs["dose_feature"]
        else:
            gene_end = self.input_gene_dim
            drug_end = gene_end + self.drug_feature_dim
            gene_features = model_inputs[:, :gene_end]
            drug_features = model_inputs[:, gene_end:drug_end]
            dose_feature = model_inputs[:, drug_end:]

        if dose_feature.ndim == 1:
            dose_feature = dose_feature.unsqueeze(1)

        return gene_features, drug_features, dose_feature

    def _forward_encoder_predictor(self, model_inputs):
        gene_features, drug_features, dose_feature = self._resolve_branch_inputs(model_inputs)
        ctx_input = torch.cat([gene_features, dose_feature], dim=1)
        drug_input = torch.cat([drug_features, dose_feature], dim=1)
        z_ctx = self.encoder_ctx(ctx_input)
        z_drug = self.encoder_drug(drug_input)
        if self.fuser_type == "film":
            z_fused = self.fuser(z_ctx, z_drug)
        else:
            z_fused = self.fuser(torch.cat([z_ctx, z_drug], dim=1))
        standardized_delta = self.predictor_head(z_fused)
        return z_ctx, z_drug, z_fused, standardized_delta

    def predict_delta(self, model_inputs):
        _, _, _, standardized_delta = self._forward_encoder_predictor(model_inputs)
        return standardized_delta * self.delta_std + self.delta_mean

    def latent_embeddings(self, model_inputs):
        z_ctx, z_drug, z_fused, _ = self._forward_encoder_predictor(model_inputs)
        return z_ctx, z_drug, z_fused

    def _adversary_logits(self, z_fused, z_drug):
        adv_input = self.grl(z_fused)
        if self.adversary_conditioning == "drug":
            # Only let the GRL act on residual plate signal not already explained by drug identity.
            adv_input = torch.cat([adv_input, z_drug.detach()], dim=1)
        return self.adversary_head(adv_input)

    def _compute_losses(self, batch, stage):
        z_ctx, z_drug, z_fused, standardized_delta_pred = self._forward_encoder_predictor(batch)
        target_delta = batch["target_delta"]
        standardized_delta_target = (target_delta - self.delta_mean) / self.delta_std

        predictor_mse_std = F.mse_loss(standardized_delta_pred, standardized_delta_target)

        if "confounder_index" in batch:
            confounder_index = batch["confounder_index"]
        elif "cell_line_index" in batch:
            confounder_index = batch["cell_line_index"]
        else:
            raise KeyError(
                "ADAEDrugResponseModule requires 'confounder_index' (or legacy "
                "'cell_line_index') in the batch; construct the DataModule with a "
                "confounder_to_index / cell_line_to_index map."
            )
        adv_logits = self._adversary_logits(z_fused, z_drug)
        adv_ce = F.cross_entropy(adv_logits, confounder_index)
        adv_acc = (adv_logits.argmax(dim=1) == confounder_index).float().mean()

        joint_loss = predictor_mse_std + adv_ce

        predicted_delta = standardized_delta_pred * self.delta_std + self.delta_mean
        predicted_expression = batch["baseline_expression"] + predicted_delta
        target_expression = batch["baseline_expression"] + target_delta
        delta_mse = F.mse_loss(predicted_delta, target_delta)
        delta_mae = F.l1_loss(predicted_delta, target_delta)
        treated_cosine = F.cosine_similarity(predicted_expression, target_expression, dim=1).mean()

        batch_size = int(target_delta.shape[0])
        prog_bar = stage != "train"
        self.log(f"{stage}_loss", joint_loss, on_step=False, on_epoch=True, prog_bar=prog_bar, batch_size=batch_size)
        self.log(f"{stage}_predictor_mse_std", predictor_mse_std, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_adv_ce", adv_ce, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_adv_acc", adv_acc, on_step=False, on_epoch=True, prog_bar=prog_bar, batch_size=batch_size)
        self.log(
            f"{stage}_lambda_adv",
            float(self.grl.alpha),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(f"{stage}_delta_mse", delta_mse, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_delta_mae", delta_mae, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            f"{stage}_treated_cosine",
            treated_cosine,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            batch_size=batch_size,
        )

        return {
            "joint_loss": joint_loss,
            "predictor_mse_std": predictor_mse_std,
            "adv_ce": adv_ce,
        }

    def _update_lambda_from_step(self, predictor_mse_std, adv_ce):
        mse_val = predictor_mse_std.detach()
        ce_val = adv_ce.detach()

        if not bool(self.ema_initialized.item()):
            self.ema_mse_std.copy_(mse_val)
            self.ema_adv_ce.copy_(ce_val)
            self.ema_initialized.fill_(True)
        else:
            beta = self.ema_momentum
            self.ema_mse_std.mul_(beta).add_(mse_val, alpha=1.0 - beta)
            self.ema_adv_ce.mul_(beta).add_(ce_val, alpha=1.0 - beta)

        ratio = (
            self.alpha_target
            * float(self.ema_mse_std.item())
            / max(float(self.ema_adv_ce.item()), self.ema_epsilon)
        )
        new_alpha = max(self.lambda_min, min(self.lambda_max, ratio))
        self.grl.alpha = float(new_alpha)

    def _shared_step(self, batch, stage):
        return self._compute_losses(batch, stage)["joint_loss"]

    def training_step(self, batch, batch_idx):
        losses = self._compute_losses(batch, "train")
        self._update_lambda_from_step(losses["predictor_mse_std"], losses["adv_ce"])
        return losses["joint_loss"]

    def validation_step(self, batch, batch_idx):
        self._compute_losses(batch, "val")

    def test_step(self, batch, batch_idx):
        self._compute_losses(batch, "test")


@torch.no_grad()
def _collect_latent_and_labels(module, dataloader, confounder_key):
    was_training = module.training
    module.eval()
    device = next(module.parameters()).device

    embeddings = []
    labels = []
    try:
        for batch in dataloader:
            if confounder_key in batch:
                label_tensor = batch[confounder_key]
            elif "confounder_index" in batch:
                label_tensor = batch["confounder_index"]
            elif "cell_line_index" in batch:
                label_tensor = batch["cell_line_index"]
            else:
                raise KeyError(
                    f"Batch is missing label key '{confounder_key}'. "
                    "Construct the DataModule with confounder_to_index or cell_line_to_index."
                )

            model_inputs = {
                "gene_features": batch["gene_features"].to(device),
                "drug_features": batch["drug_features"].to(device),
                "dose_feature": batch["dose_feature"].to(device),
            }
            _, _, z_fused = module.latent_embeddings(model_inputs)
            embeddings.append(z_fused.detach().cpu().to(torch.float32).numpy())
            labels.append(label_tensor.detach().cpu().to(torch.long).numpy())
    finally:
        if was_training:
            module.train()

    import numpy as _np

    return _np.concatenate(embeddings, axis=0), _np.concatenate(labels, axis=0)


def probe_latent_plate_accuracy(
    module,
    fit_dataloader,
    eval_dataloader=None,
    confounder_key="confounder_index",
    max_iter=1000,
    C=1.0,
    test_size=0.2,
    random_state=0,
):
    """Fit a linear (sklearn LogisticRegression) probe on frozen z_fused embeddings.

    Quantifies how linearly decodable the confounder (e.g. plate) is from the ADAE
    fused latent. If ``eval_dataloader`` is provided, the probe is fit on
    ``fit_dataloader`` embeddings and evaluated on ``eval_dataloader`` embeddings;
    otherwise the fit embeddings are split internally with ``test_size``.

    Returns a dict with the probe accuracy alongside random and majority-class
    baselines so the notebook can decide whether the latent actually carries
    confounder information and whether the adversary reduced it.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    fit_embeddings, fit_labels = _collect_latent_and_labels(
        module, fit_dataloader, confounder_key
    )
    if fit_embeddings.shape[0] == 0:
        raise ValueError("fit_dataloader produced zero samples for the latent probe.")

    if eval_dataloader is not None:
        train_embeddings, train_labels = fit_embeddings, fit_labels
        eval_embeddings, eval_labels = _collect_latent_and_labels(
            module, eval_dataloader, confounder_key
        )
        if eval_embeddings.shape[0] == 0:
            raise ValueError("eval_dataloader produced zero samples for the latent probe.")
    else:
        from sklearn.model_selection import train_test_split

        unique_fit_classes, fit_class_counts = np.unique(fit_labels, return_counts=True)
        can_stratify = fit_class_counts.min() >= 2 and len(unique_fit_classes) >= 2
        train_embeddings, eval_embeddings, train_labels, eval_labels = train_test_split(
            fit_embeddings,
            fit_labels,
            test_size=float(test_size),
            random_state=int(random_state),
            stratify=fit_labels if can_stratify else None,
        )

    unique_train_classes = np.unique(train_labels)
    if len(unique_train_classes) < 2:
        raise ValueError(
            "Latent probe needs at least 2 confounder classes in the training split; "
            f"got {len(unique_train_classes)}."
        )

    probe = LogisticRegression(
        C=float(C),
        max_iter=int(max_iter),
        solver="lbfgs",
        random_state=int(random_state),
    )
    probe.fit(train_embeddings, train_labels)
    probe_accuracy = float(probe.score(eval_embeddings, eval_labels))

    train_class_counts = np.bincount(train_labels)
    train_majority_class = int(train_class_counts.argmax())
    majority_baseline = float(np.mean(eval_labels == train_majority_class))
    n_eval_classes = int(np.unique(eval_labels).shape[0])
    random_baseline = 1.0 / float(max(n_eval_classes, 1))

    return {
        "probe_accuracy": probe_accuracy,
        "random_baseline": random_baseline,
        "majority_baseline": majority_baseline,
        "n_train_samples": int(train_embeddings.shape[0]),
        "n_eval_samples": int(eval_embeddings.shape[0]),
        "n_train_classes": int(unique_train_classes.shape[0]),
        "n_eval_classes": n_eval_classes,
        "latent_dim": int(train_embeddings.shape[1]),
    }


MODEL_REGISTRY = {
    "ridge": RidgeDrugResponseModule,
    "mlp": MLPDrugResponseModule,
    "adae": ADAEDrugResponseModule,
}


def build_model(model_name, **model_kwargs):
    normalized_model_name = str(model_name).strip().lower()
    if normalized_model_name not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported MODEL_NAME: {model_name}. Available models: {available_models}")
    return MODEL_REGISTRY[normalized_model_name](**model_kwargs)
