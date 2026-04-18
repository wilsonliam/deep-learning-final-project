# AD-AE Architecture Context

This document distills the architecture, training loop, and data assumptions of the **Adversarial Deconfounding Autoencoder (AD-AE)** (Dincer et al., 2020) as implemented in this repository, and maps each piece to an adaptation target:

- **Input**: 2048-bit Morgan fingerprints (binary chemical structure descriptors)
- **Output**: delta expression vector for 1000 genes (e.g. `treated − control` on pseudobulked scRNA-seq)
- **Goal of the port**: keep the adversarial deconfounding idea but convert the self-supervised autoencoder into a supervised fingerprint→Δexpression regressor.

The downstream agent is assumed to know the user's own codebase better; this file's job is to be the faithful spec of *this* repo's model so it can be ported, not re-derived.

---

## 1. TL;DR

AD-AE trains an autoencoder on gene-expression `X` jointly with an adversary that tries to predict a confounder `Z` (batch, sex, age, etc.) from the latent code. The autoencoder is optimized to **minimize reconstruction error while maximizing adversary error**, producing latent embeddings that encode biology but not the confounder.

Combined objective used by the autoencoder step:

```
L_total = MSE(x, x_hat) − λ · CE(z_hat, z)
```

implemented in Keras by compiling a two-output model with `loss_weights=[1.0, -λ]`. Training is alternating min–max: freeze AE, train adversary one epoch; freeze adversary, train AE on a minibatch with the combined loss; repeat for `T_iter` iterations.

---

## 2. Architecture

Two reference implementations exist in the repo. Both share the same skeleton; the adversary head differs based on whether the confounder is multi-class or binary.

### 2.1 Autoencoder (encoder + decoder)

From [KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py](KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py) lines 117–143:

```117:143:KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py
def _create_autoencoder_net(self, inputs, n_features, latent_dim):

    #Encoder
    dense1 = Dense(500, activation='relu')(inputs)
    dropout1 = Dropout(0.1)(dense1)
    latent_layer = Dense(latent_dim)(dropout1)

    #Decoder
    dense2 = Dense(500, activation='relu')
    dropout2 = Dropout(0.1)
    outputs = Dense(n_features)

    decoded = dense2(latent_layer)
    decoded = dropout2(decoded)
    decoded = outputs(decoded)
```

- **Encoder**: `Input(n_features) → Dense(500, relu) → Dropout(0.1) → Dense(latent_dim)` (linear bottleneck, no activation).
- **Decoder**: `Dense(500, relu) → Dropout(0.1) → Dense(n_features)` (linear output).
- The decoder is built twice: once wired into the full AE graph, and once as a standalone `Model(decoder_input, decoded)` so an embedding can be decoded independently.
- The TCGA variant is identical but with `Dropout(0.0)` (see [TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Sex.py](TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Sex.py) lines 120–146).

### 2.2 Adversary

Multi-class (5 batches) variant ([KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py](KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py) lines 146–150):

```146:150:KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py
def _create_adv_net(self, inputs):
    dense1 = Dense(100, activation='relu')(inputs)
    dense2 = Dense(100, activation='relu')(dense1)
    outputs = Dense(5, activation='softmax')(dense2)
    return Model(inputs=[inputs], outputs = [outputs],  name = 'adversary')
```

Binary (sex) variant ([TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Sex.py](TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Sex.py) lines 149–153):

```149:153:TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Sex.py
def _create_adv_net(self, inputs):
    dense1 = Dense(50, activation='relu')(inputs)
    dense2 = Dense(50, activation='relu')(dense1)
    outputs = Dense(1, activation='sigmoid')(dense2)
    return Model(inputs=[inputs], outputs = [outputs],  name = 'adversary')
```

- Adversary input is the **latent code**, not the raw features.
- Capacity: two hidden ReLU layers, 50–100 units each. Output head matches the confounder type (softmax/CE or sigmoid/BCE).

### 2.3 Default hyperparameters observed in scripts

| Setting | BRCA script | TCGA (sex) script |
|---|---|---|
| `latent_dim` | 100 | 50 |
| `lambda_val` | 0.1 | 0.1 |
| Optimizer | Adam (Keras default lr) | Adam |
| `batch_size` | 128 | 128 |
| Pretrain epochs | 10 | 5 |
| `T_iter` (adv phase) | 200 | 3000 |
| Dropout | 0.1 | 0.0 |

---

## 3. Training loop

Three Keras models are compiled from shared layers:

1. `self._ae` — inputs → decoded, MSE loss. Used for pretraining the autoencoder and for standalone reconstruction metrics.
2. `self._adv` — inputs → encoder → adversary, CE/BCE loss. Used to train the adversary **through the frozen encoder**.
3. `self._ae_w_adv` — inputs → [decoded, adversary_prediction], dual loss with weights `[1.0, -λ]`. Used for the AE min–max step.

The combined model is compiled with a negative adversary weight (from [KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py](KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py) lines 179–189):

```179:189:KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py
def _compile_ae_w_adv(self, inputs, ae_net, encoder_net, adv_net):
    ae_w_adv = Model(inputs=[inputs], outputs = [ae_net(inputs)] + [adv_net(encoder_net(inputs))])
    self._trainable_ae_net(True) #classifier is trainable
    self._trainable_adv_net(False) #Freeze the adversary
    loss_weights = [1., -1 * self.lambda_val] #classifier loss - adversarial loss
    #Now compile the model with two losses and defined weights
    ae_w_adv.compile(loss=['mse', 'categorical_crossentropy'],
                      metrics=['mse', 'accuracy'],
                      loss_weights=loss_weights,
                      optimizer='adam')
    return ae_w_adv
```

Freezing uses a closure factory (`_make_trainable`) that toggles `trainable` on every layer of a submodel.

### 3.1 Pretrain

```202:213:KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py
def pretrain(self, x, z, validation_data=None, epochs=10):
    self._trainable_ae_net(True)
    self._ae.fit(x.values, x.values, epochs=epochs)
    self._trainable_ae_net(False)
    self._trainable_adv_net(True)

    if validation_data is not None:
        x_val, z_val = validation_data

    self._adv.fit(x.values, z.values,
                  validation_data = (x_val.values, z_val.values),
                    epochs=epochs, verbose=2)
```

- Pretrain the AE on reconstruction.
- Freeze AE, pretrain the adversary on latent→Z.

### 3.2 Alternating min–max

```224:257:KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py
for idx in range(T_iter):
    ...
    # train adversary
    self._trainable_ae_net(False)
    self._trainable_adv_net(True)
    history = self._adv.fit(x.values, z.values,
                            validation_data = (x_val.values, z_val.values),
                            batch_size=batch_size, epochs=1, verbose=1)
    ...
    # train autoencoder
    self._trainable_ae_net(True)
    self._trainable_adv_net(False)
    indices = np.random.permutation(len(x))[:batch_size]
    history = self._ae_w_adv.fit(x.values[indices],
                             [x.values[indices]] + [z.values[indices]],
                             batch_size=batch_size, epochs=1, verbose=1,
                             validation_data = (x_val.values,
                             [x_val.values] + [z_val.values]))
```

Key asymmetry: the adversary is trained one full epoch on the whole training set per iteration; the AE is trained on a single random 128-sample minibatch per iteration. This biases the adversary to converge quickly before each AE step.

---

## 4. Data pipeline assumptions

- Inputs are **samples × genes** dataframes (indexed by sample id) pre-reduced by k-means clustering to 1000 (BRCA) or 500 (TCGA) "meta-gene" features. See [KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py](KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py) lines 31–36.
- Confounder `Z` is a one-hot dataframe for multi-class (BRCA batch labels, [KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py](KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py) lines 38–47) or a 0/1 dataframe for binary ([TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Sex.py](TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Sex.py) lines 36–45).
- Preprocessing: `sklearn.StandardScaler` **fit on train only**, then applied to test (lines 52–56 of the BRCA script).
- Splits: a leave-one-dataset-out protocol in BRCA (lines 386–410) in addition to the 80/20 random split; `random_state=12345`.
- Random seeds are multiplicative per fold: `seed(123456 * run)` and `set_random_seed(123456 * run)` (lines 67–68).

Outputs persisted after training:

- Full-matrix embeddings: `encoder.predict(X)` saved as `ADV_Embedding_*.tsv` (e.g. lines 441–443).
- Encoder + decoder weights as JSON + H5 files under `ADV_FILES*/` folders.

---

## 5. Adaptation to fingerprint → Δexpression

### 5.1 Conceptual shift

AD-AE is **self-supervised** (`X → X`) with an adversary on the latent. The user's problem is **supervised regression** (`fingerprint → Δexpression`). The adaptation replaces the reconstruction head with a prediction head but keeps the adversarial deconfounding apparatus around a latent bottleneck.

Proposed topology:

```
fp (2048 binary)
    → Dense(1024, relu) → Dropout
    → Dense(512,  relu) → Dropout
    → Dense(latent_dim)          # linear bottleneck, adversary attacks here
    → Dense(512,  relu) → Dropout
    → Dense(1000)                 # linear Δ-expression head
```

- `encoder` = fp → latent. `decoder` → `predictor`: latent → Δexpression. `_ae` → `_predictor`. `_ae_w_adv` → `_predictor_w_adv`.
- Loss on the prediction head: MSE (optionally Huber or `1 − Pearson`). Adversary head stays CE/BCE/MSE depending on confounder type.
- Combined training loss: `MSE(ŷ, Δ) − λ · L_adv(ẑ, z)`.

### 5.2 Reusable vs must-change

**Reusable as-is:**

- `_make_trainable` freezing pattern (lines 109–114 of the BRCA script).
- Three-model compile pattern (`_predictor`, `_adv`, `_predictor_w_adv`) with `loss_weights=[1.0, -λ]`.
- Pretrain-then-alternate schedule (`pretrain` then `fit` with `T_iter`).
- `StandardScaler` fit-on-train convention for the Δexpression target; fingerprints should stay binary and **not** be standardized.
- Multiplicative seed scheme for fold reproducibility.

**Must change:**

- **Task / loss**: reconstruction MSE on `X` becomes supervised MSE on Δexpression. `self._ae.fit(x, x)` becomes `self._predictor.fit(fp, delta)`.
- **Input shape**: `n_features = 2048`. First dense layer should be wider (e.g. 1024) because fingerprints are sparse binary.
- **Output shape**: `n_outputs = 1000` continuous values; keep the final dense linear (no activation) and loss `mse`.
- **Confounder choice**: in pseudobulk scRNA, candidates for `Z` are `batch/plate`, `donor`, `cell_type`, or continuous QC covariates (library size, % mito, number of cells in the pseudobulk). Match activation + loss:
  - multi-class (donor, plate, cell_type): `softmax` + `categorical_crossentropy` (one-hot `Z`)
  - binary: `sigmoid` + `binary_crossentropy`
  - continuous: linear + `mse`
- **Pseudobulk data assembly**: rows are `(compound, cell_type, donor/batch)` aggregates. Compute Δ as `mean_expr(treated) − mean_expr(matched_control)` per `(cell_type, donor)` group so you hold the biological baseline fixed per unit. Carry the `(cell_type, donor/batch)` identifiers into the confounder matrix.
- **Splitting**: use a **compound-level split** (ideally scaffold split) rather than a random row split — random splits leak, because the same compound can appear across many `(cell_type, donor)` rows.
- **API porting**: the repo uses Keras 1.x on TF1 (`from tensorflow import set_random_seed`, `keras.layers`). Port to TF2 (`tf.random.set_seed`, `tf.keras`) or PyTorch. The min–max pattern in PyTorch is usually expressed with a Gradient Reversal Layer (GRL) in a single forward pass instead of three separate compiled models; either approach is valid — see §7.

### 5.3 Hyperparameter starting points to carry over

- `lambda_val = 0.1` (then sweep 0.01 – 1.0).
- `latent_dim = 64` (range 32–128; BRCA used 100, TCGA used 50).
- Pretrain: 5–10 epochs predictor, then 5–10 epochs adversary.
- Adversarial phase: start at `T_iter = 200` and scale up only if the adversary is losing accuracy too quickly.
- Adversary capacity: 2 hidden layers of 50–100 units.
- Dropout: 0.1 in the predictor; keep adversary dropout at 0.
- Batch size: 128; Adam default lr (1e-3).

---

## 6. Drop-in variable mapping

- `X` (expression matrix, samples × genes) → `fingerprints` (samples × 2048 binary)
- `Z` (confounder one-hot/binary) → `confounder` (batch / donor / cell_type per pseudobulk row)
- target for the "AE" head: `X` itself → `delta_expression` (samples × 1000)
- `n_features` (input width) → 2048 (fingerprint dim)
- `n_outputs` (new concept; was `n_features` in the decoder) → 1000 (Δexpression dim)
- `latent_dim` → `latent_dim` (unchanged concept)
- `self._ae` → `self._predictor`
- `self._encoder` → `self._encoder` (still useful for downstream embeddings)
- `self._decoder` → `self._head` or `self._predictor_head`
- `self._adv` → `self._adv`
- `self._ae_w_adv` → `self._predictor_w_adv`
- `loss=['mse', 'categorical_crossentropy']` → `loss=['mse', <confounder loss>]`
- `loss_weights=[1., -lambda_val]` → unchanged
- `StandardScaler` on `X` → `StandardScaler` on `delta_expression` only; fingerprints stay as `float32` binary.
- `samples_list` leave-one-dataset-out → leave-one-`(cell_type | donor | scaffold)`-out depending on the evaluation you want.

---

## 7. Gotchas taken straight from the existing code

- **Negative loss weight implements min–max.** `loss_weights = [1., -1 * self.lambda_val]` inside `_compile_ae_w_adv` is what causes the AE gradient to *hurt* the adversary. Forgetting the minus sign turns AD-AE into a regular multi-task model and defeats the whole point.
- **Adversary vs AE training asymmetry.** Each iter the adversary sees the full training set; the AE sees one random 128-sample minibatch (`indices = np.random.permutation(len(x))[:batch_size]`). Preserve this asymmetry unless you have a reason not to — it is part of why the adversary stays strong enough to provide a useful gradient.
- **Two decoders, shared weights.** The decoder layers are created as standalone layer objects then wired twice: once as part of the AE graph and once via `decoder_input = Input(shape=(latent_dim,))`. This is how the repo reuses the same decoder weights for both inline decoding and standalone `decoder.predict(latent)`. If you port to PyTorch, just reuse the same `nn.Sequential` in two forward methods.
- **Seeds are multiplicative, not additive.** `seed(123456 * run)` and `set_random_seed(123456 * run)`. `run=0` collapses all seeds to 0 — use `run ≥ 1` for fold reproducibility.
- **Dropout rate differs across datasets.** BRCA uses 0.1, TCGA uses 0.0. Tune per dataset size; with pseudobulk scRNA sample counts often in the low thousands, a nonzero dropout (0.1–0.3) in the predictor is reasonable.
- **Keras 1.x API.** `from tensorflow import set_random_seed`, `keras` (not `tf.keras`), `Dense(...)` without `input_shape`, `model.fit` with positional `x, y`. You will rewrite these calls for whatever framework the user's codebase uses.
- **Class imbalance is not handled in the adversary loss here.** `compute_class_weight` is imported but not used. If your confounder is imbalanced (e.g. one dominant donor), add `class_weight` to `self._adv.fit` or switch to focal loss.
- **Validation data is required by `fit`.** The loop reads `x_val, z_val` unconditionally inside the iteration body. Always pass `validation_data` when porting, or strip those branches.

---

## 8. Files worth reading in order

1. [KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py](KMPLOT_BRCA_EXPRESSION/Adversarial_Deconfounder_AE_Generate_Embeddings.py) — canonical multi-class implementation, lines 62–383 are the full `AdversarialDeconfoundingAutoencoder` class.
2. [TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Sex.py](TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Sex.py) — binary-confounder variant, same class with sigmoid head.
3. [TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Age.py](TCGA_BRAIN_EXPRESSION/Adversarial_Deconfounding_AE_Generate_Embeddings_Age.py) — continuous-confounder variant (useful template if you want to adversarially deconfound e.g. library size).
4. [README.md](README.md) — paper-level framing and folder layout.

## 9. Out of scope for this document

- Running or modifying any of the existing notebooks/scripts.
- Writing the adapted fingerprint→Δexpression training code itself — that is the next agent's job, given full context of the user's own codebase.
