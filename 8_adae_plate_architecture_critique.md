# Architecture & Design Critique — `8_adae_plate.ipynb`

Scope: AD-AE plate-adversary notebook. Critique is ordered by expected impact on
predictor quality and on the defensibility of the deconfounding claim. Each
section states the concern, why it matters, and concrete changes to try.

---

## 1. Plate identity is confounded with drug under `drug_blind` — interpret with care

**Concern.** Under `SPLIT_MODE="drug_blind"`, entire drugs are held out of train,
but the notebook also requires val/test plates to be a **subset of train
plates** (unseen plates are dropped with a warning). That means each plate in
train carries a specific mixture of drugs, and the adversary is learning a
plate signature that is partly drug-structure (which drugs co-load on that
plate) and partly technical (reagents, run day, edge effects). When the GRL
pushes the encoder to suppress that plate signal, it also partially suppresses
drug-class signatures that are plate-correlated — which directly competes with
the predictor on held-out drugs whose nearest-neighbor drugs live on specific
plate subsets.

**Why it matters.** A "successful" drop in plate decodability may be partially
a drop in drug generalization. This undermines the interpretation of the whole
main-vs-ablation comparison.

**Suggestions.**
- Before training, audit per-plate variance: how much of plate-level signal is
  explained by the drug set vs a known technical covariate (row/column
  position, run date, operator if available)? If >30–40% of plate variance
  aligns with drug-set composition, the confounding is real.
- **Condition the adversary**: predict plate from `(z_fused, drug_embedding)`
  rather than `z_fused` alone. This forces the adversary to use only residual
  plate information not attributable to the drug, so the GRL gradient only
  removes the technical part.
- Alternatively, stratify: restrict the adversary's CE to within-drug
  comparisons (group the CE by drug and average), so the gradient is driven by
  plate differences among the same drug, not across drugs.

---

## 2. Encoder capacity is badly imbalanced; fusion is too simple

**Concern.** `CTX_BRANCH_HIDDEN_DIMS=(2048, 1024)` with `Z_CTX_DIM=128` vs
`DRUG_BRANCH_HIDDEN_DIMS=(512, 128)` with `Z_DRUG_DIM=64` gives the context
branch roughly an order of magnitude more parameters than the drug branch, and
concatenation produces a `z_fused` input with twice as much context signal as
drug signal (128:64).

**Why it matters.** For drug-response prediction, the drug is the high-signal
channel — most of the information gain over "predict zero Δ" comes from which
compound is applied at what dose. A structural 2:1 bias toward context
suppresses drug sensitivity. The 1024→128 bottleneck in the ctx branch is also
aggressive; if input_gene_dim ≈ 978 (L1000), the 2048 expansion wastes
parameters before the hard crunch.

**Suggestions.**
- Balance latent dims: e.g., `Z_CTX_DIM=96, Z_DRUG_DIM=96` or even flip to
  `64 / 128` to bias toward drug.
- Shrink the ctx branch: `(1024, 512)` or just `(1024,)` is likely sufficient.
- Replace concat-then-MLP fusion with **FiLM** (feature-wise linear
  modulation) or bilinear gating: have `z_drug` produce per-feature scale and
  shift parameters that modulate `z_ctx`. This encodes the inductive bias
  drug-response models actually want — "same cell, different drug, different
  response" — as a structural property, not something the fuser has to
  discover.

---

## 3. The predictor should see the baseline directly

**Concern.** The predictor only sees `z_fused`, which was produced by
`Enc_ctx` abstracting away from the raw DMSO baseline. For per-gene Δ
prediction, many Δ patterns are partially multiplicative in the baseline
(genes that are off in baseline can't go further off).

**Why it matters.** Forcing the predictor to reconstruct baseline-dependent
structure from a 128-dim abstraction is wasteful.

**Suggestions.**
- Add a skip connection: concatenate the normalized baseline expression into
  the predictor's input, or use a gene-wise residual form
  `Δ̂ = MLP(z_fused) + W · baseline`. This is the Compose-style pattern and it
  is cheap. It does not touch the adversarial machinery.
- If worried about the adversary getting a free ride through the skip, gate
  the skip via `z_drug` so the baseline contribution is still conditioned on
  the compound.

---

## 4. Adversary pressure and scheduling

**Concern 4a — adversary capacity.** `ADV_HIDDEN_DIMS=(100, 100)` may be too
weak relative to a 256-dim `z_fused` input and potentially dozens of plate
classes. A weak adversary gives the encoder a false sense of invariance:
`val_adv_acc` near random can mean the encoder removed plate, or it can mean
the adversary was never strong enough to find it.

**Concern 4b — no warmup.** `LAMBDA_MIN=1e-3` is small but nonzero, and the
dynamic-λ formula engages immediately. The encoder starts being pulled by the
adversary before it has learned anything useful.

**Concern 4c — single optimizer, shared learning rate.** One AdamW with
`lr=1e-3` trains encoder, predictor, fuser, and adversary together. Standard
DANN practice uses a higher adversary learning rate, and sometimes inner loops
(`k` adversary steps per encoder step), to keep the adversary near-optimal.
The sanity checks verify gradient direction but not whether the adversary is
near-optimal at each encoder step.

**Suggestions.**
- Scale adversary: at minimum `(256, 256)`, or `(4 * z_fused_dim,
  4 * z_fused_dim)`, or scale with `log(n_plates)`.
- Warmup: linear ramp of λ from 0 over the first 2–5 epochs, or the classic
  DANN schedule `λ = 2/(1+exp(-10·p)) − 1` where `p ∈ [0, 1]` is training
  progress. Keep the dynamic-λ formula as the steady-state rule but gate it
  behind a warmup.
- Either give the adversary parameters a separate param group with a
  higher learning rate, or run `k=2–5` adversary steps per encoder step.

---

## 5. Dose signal is likely drowned

**Concern.** A single scalar dose concatenated with a ~2048-bit Morgan
fingerprint has negligible influence on `z_drug` — the dose coefficient is one
column out of ~2049 in the first Linear.

**Why it matters.** Dose-response is a core axis of drug response. If it's
effectively ignored, the model collapses across concentrations.

**Suggestions.**
- Use `log10(dose + ε)` rather than raw dose (dose distributions are
  lognormal).
- Expand dose into a small learned embedding or Fourier features (8–16 dims)
  before concat, so it has comparable "representation mass" to the
  fingerprint.
- Inject dose as a FiLM conditioner on `z_drug` rather than concatenating at
  the raw input, so dose scales the whole drug representation.

---

## 6. Hidden design choices not exposed as hyperparameters

**Concern.** Activation, dropout, and normalization are not in the knob list
but they matter here.

**Specifically: BatchNorm anywhere upstream of `z_fused` interacts badly with
the plate adversary.** If batches are plate-imbalanced (very likely — plates
partition drugs), BN statistics leak plate information in a way that the GRL
cannot undo (BN computes per-batch statistics outside the autograd path that
the adversary operates on). This can silently cap how invariant `z_fused` can
become.

**Suggestions.**
- Switch any BN layers to **LayerNorm** or **GroupNorm** on the ctx/drug
  branches and the fuser.
- Expose dropout in the ctx branch (which is the high-capacity branch most at
  risk of overfitting plate structure) and expose the activation choice. GELU
  or SiLU tend to outperform ReLU for this kind of MLP.

---

## 7. The headline claim rests on a single-seed comparison

**Concern.** `L.seed_everything(RANDOM_SEED)` is called before the ablation,
which is good for reproducibility of that single pair of runs. But per-plate
`var_of_plate_means`, `treated_cosine` variance, and probe accuracy across a
single seed pair are noisy enough that the "main vs ablation" delta is
plausibly within seed variance.

**Why it matters.** This is the central empirical claim of the notebook. It
needs statistical footing.

**Suggestions.**
- Run {main, ablation} over **3–5 seeds** and report mean ± CI on the
  comparison metrics. If compute is tight, even 3 seeds changes the
  interpretation.
- Add a **non-linear probe** (small MLP or kNN) alongside the linear
  `LogisticRegression` probe. A linear probe can show plate is "gone" while a
  non-linear probe decodes it fine — meaning the encoder has only rotated
  plate structure out of the linear subspace, not removed it.

---

## 8. Smaller items worth a pass

- **EMA momentum.** `EMA_MOMENTUM=0.99` at `BATCH_SIZE=512` gives a
  ~100-batch effective averaging window. On smaller train sets this can make
  the λ feedback loop sluggish. Sanity-check the effective adaptation
  timescale against steps-per-epoch; 0.9–0.95 may be more responsive.
- **Early-stopping monitor.** `val_treated_cosine` can peak mid-oscillation
  of the saddle. Consider a compound monitor like
  `val_treated_cosine − β · max(0, val_adv_acc − random_baseline)` so
  checkpoints are penalized for being weakly deconfounded at a local cosine
  peak.
- **λ ceiling.** `LAMBDA_MAX=10.0` is high. If CE briefly collapses (e.g.,
  the adversary gets very confident early because the encoder hasn't moved
  yet), the ratio can spike toward 10, destabilizing the encoder. The init
  sanity check catches initial clipping but not mid-training spikes. A
  smoother cap — e.g., a softplus on the pre-clamp value — or a tighter
  ceiling like 3–5 is worth trying.
- **Probe fit discipline.** Confirm the linear probe is fit on train `z_fused`
  and evaluated on test `z_fused`, not accidentally fit on test. The code
  looks right but this number is central and worth a re-read.
- **Plate as categorical.** Plates are treated as unordered classes via CE.
  If plates have spatiotemporal structure (run order, operator, reagent lot),
  that structure is unused. A hierarchical adversary (predict plate group
  first, then within-group) or a continuous plate embedding could yield more
  informative pressure. Low priority unless you have plate metadata.
- **Adversary robustness.** Consider re-initializing the adversary head
  periodically (every N epochs) or training an ensemble of adversaries. This
  mitigates the "encoder learns to fool one specific classifier" failure mode
  and makes the deconfounding claim stronger.

---

## Priority order for changes

If only two changes are made first:

1. **Add a baseline skip into the predictor** (Section 3). Cheap, unlikely to
   hurt, likely to help predictor quality materially.
2. **Re-run main vs ablation over ≥3 seeds with a non-linear probe**
   (Section 7). This is the minimum bar to make the deconfounding claim
   defensible.

If compute allows a third:

3. **Rebalance encoder capacity and switch to FiLM fusion** (Section 2).
   Larger structural change but directly attacks the drug-response inductive
   bias.

Items 1 and 4b (warmup) are near-free and should probably go in regardless.
Item 6 (BatchNorm audit) should be checked immediately — if BN is present in
the ctx branch, that alone could be capping the adversary's effectiveness.
