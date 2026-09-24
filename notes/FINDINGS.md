# What was established before this repo existed

Measured on a laptop (i5-1145G7, 4 physical cores, CPU-only torch 2.14),
2026-09-24. These are the facts the design of this repo rests on.

## The reference numbers to beat

Same Dreyer2023 split, same benchopt objective, so directly comparable:

| solver | test balanced accuracy |
|---|---|
| `EEGNet` (competition's shipped baseline, 2,420 params) | 0.7651 |
| `MyModel` (EEGNet + per-window z-score, AdamW/cosine) | 0.7794 |
| chance | 0.50 |

## The shipped NeuroTTT checkpoints are imagined speech, not motor imagery

`neuro-ttt/CBraMod/model_weights/{baseline,neurottt}.pth` are the BCIC2020-III
models from `step2_finetune.ipynb`, **not** BCIC-IV-2a MI models. From the state
dicts: `classifier.1.weight` is `(600, 38400)` = 64 ch x 3 patches x 200, and the
final layer is `(5, 200)`. `neurottt.pth` additionally carries `ssl_band` (5-way)
and `ssl_amp` (4-way) heads.

Dreyer is 27 ch / 4 patches / 2 classes, so those classifiers cannot be reused.
Only the backbone transfers. `weights/cbramod_speech_*.pth` here are the
`backbone.*` keys only (4.88M params, ~20 MB) — the 112 MB originals are over
GitHub's 100 MB file limit and 23M of those params were a head we cannot use.

**There is no pretrained NeuroTTT to evaluate.** NeuroTTT is a recipe, not a
model: Stage I is a supervised + SSL fine-tuning stage by construction
(`L_main` needs labels). Only Stage II (Tent) is training-free, and it applies
on top of any arm — including the frozen-backbone probe.

## What the paper actually prescribes (read before changing the SSL code)

From the PDF (arXiv:2509.26301v2), §3.1, §3.2, §4.1 and Appendix A.1/B.4:

- **Stage I SSL is a *pair* of heads, and the second one is task-specific.**
  Stopped-Band Prediction is shared; imagined speech adds Amplitude Scaling,
  mental stress adds Anterior–Posterior Flip, and **motor imagery adds Temporal
  Jigsaw** (split the trial into 2–3 consecutive chunks, shuffle, predict the
  order). Band divisions differ per task too — MI is Table 6: theta 3–7,
  mu 8–13, beta 13–30, low gamma 30–45.
- **Loss weights differ per task** (Table 7). Motor imagery is
  `w_band = 0.1`, `w_jigsaw = 0.8` — the jigsaw term dominates. The reference
  notebook's 0.1/0.1 is the mental-stress row; imagined speech is 0.6/0.6.
- **Stage II has two variants**, and Tent is only one of them:
  (a) *TTT with SSL* — one full-parameter Adam step (lr 1e-5) on the SSL loss
  per unlabeled test sample, batch size 1, then the model is **reset to its
  original weights before the next sample** (`online=False`). This is the
  per-sample/subject calibration.
  (b) *TTT with Tent* — entropy minimisation, normalisation parameters only,
  several updates per batch, online.
- **The backbone is never frozen.** §4.1: "In all of our finetuning and
  adaptation settings, the foundation model remains fully trainable." LoRA,
  which does freeze it, is the *worst* method on MI (0.4164). The paper's
  explanation: "the frozen backbone's representations remain largely
  unchanged ... a small low-rank tweak may be insufficient to bridge the gap."
  So this repo's `probe` arm is a diagnostic floor, not a contender.
- **MI protocol**: BCIC-IV-2a cross-subject, subjects 1–5 train / 6–7 val /
  8–9 test; 22 ch, 250 Hz resampled to 200 Hz; Adam lr 1e-4; converges within
  20 epochs. Both backbones "internally downsample input signals to 200 Hz",
  which is why this repo resamples Dreyer 120 -> 200 Hz.
- **CBraMod has no BatchNorm.** The paper writes Tent as updating BN
  statistics; CBraMod is LayerNorm throughout, so Tent adapts 24 LayerNorm
  affines = 9,600 of 4.99M params (0.19%).

Paper Table 3 (BCIC-IV-2a), CBraMod column, for reference:
Linear 0.5028 | Shallow MLP 0.4578 | LoRA 0.4164 | SHOT 0.5354 |
TTT w/ SSL 0.5435 | TTT w/ Tent 0.5674.

## Published numbers for context

NeuroTTT (arXiv:2509.26301) benchmarks motor imagery on BCI Competition IV-2a —
the same dataset as this repo's `tangermann2012` smoke study:

| method | CBraMod | LaBraM |
|---|---|---|
| full supervised fine-tuning | 0.5028 | 0.4111 |
| + TTT w/ SSL | 0.5435 | 0.4384 |
| + Tent (NeuroTTT) | 0.5674 | 0.4511 |

A plain 2,420-parameter EEGNet scored **0.5825** on that dataset here (different
split, so not a clean head-to-head — but the direction matches
arXiv:2507.01196, which finds EEG foundation models repeatedly failing to beat
compact supervised CNNs on MI). Expect the foundation-model arms to start
*behind* EEGNet. The interesting question is whether Tent closes the gap.

## Measured cost (laptop CPU; the GPU is 1-2 orders faster)

CBraMod backbone on `(B, 27, 4, 200)`: forward **6.4 ms/window**,
forward+backward **33 ms/window**. Dreyer splits: train 12,392 / val 3,360 /
test 5,040 windows.

| operation | laptop CPU |
|---|---|
| full test pass | ~32 s |
| cache backbone features, all splits (1.8 GB) | ~2.2 min |
| head-only training on cached features | 3.4 s/epoch |
| full fine-tune | ~6.5 min/epoch |
| NeuroTTT stage I (3x forward) | ~20 min/epoch |
| Tent over the test set | ~3.5 min |

`OMP_NUM_THREADS=8` is **not** faster than 4 on that laptop (4 physical cores).

## Two correctness issues found and fixed

- **Validation must be grouped by subject.** The random window split in the
  earlier `my_model.py` leaks across subjects: it read 0.8674 against a real
  0.7794 on Dreyer, a ~9-point optimism gap. Warm-up is scored on unseen
  subjects and the sealed phase on unseen sessions, so `_split_by_subject`
  holds out whole subjects.
- **A training run must ignore `weights.pt`.** Every benchopt parameter
  configuration shares `outputs/<solver-name>/` as its submission dir, so
  honouring a checkpoint there silently warm-starts one arm from another arm's
  weights. `run/matrix.sh` additionally gives each arm its own
  `COMPET_SUBMISSION_DIR`.

## Input adaptation, verified

Dreyer windows are `(B, 27, 480)` @ 120 Hz; CBraMod wants 1 s patches @ 200 Hz.
`cbramod/core.py:to_patches` FFT-resamples 480 -> 800 and views the result as
`(B, 27, 4, 200)` — matching CBraMod's own `preprocessing_bciciv2a.py`
(`resample(sample, 800)` then `reshape(22, 4, 200)`). The torch implementation
agrees with `scipy.signal.resample` to 2e-15.

CBraMod was pretrained on uV/100 (`bciciv2a_dataset.py` returns `data/100`)
while Dreyer arrives RobustScaler'd (std 0.71, range [-8.9, 4.1]). Close, but
not identical — hence `gain=` is a swept input scale, and stage 0 of the matrix
exists to pin it down on the cheap frozen-backbone arm.

## REVE, checked 2026-09-24

- `brain-bzh/reve-base` (69M params, embed 512) and `brain-bzh/reve-large`
  (400M, embed 1216, 19 heads, depth 22) are on the Hub and, despite what
  braindecode's docstring says, **not gated** — anonymous download returns 200.
  1.56 GB for large, 0.28 GB for base, so neither is vendored here.
- `REVE.from_pretrained(repo, n_outputs, n_chans, n_times, sfreq, chs_info)`
  works, and `forward(x, return_features=True)["features"]` gives
  `(B, C, S, D)`. All 27 Dreyer channel names resolve in the position bank to
  `(27, 3)` coordinates.
- Patching is `patch_size=200, patch_overlap=20` (stride 180), so an 800-sample
  window yields 4 patches — the same count as CBraMod, by coincidence.
- **The position bank downloads `positions.json` from the Hub at model
  construction**, not just the weights. Warm `HF_HOME` on a login node.
- Measured on the laptop CPU: reve-base **51 ms/window** forward, ~8x CBraMod's
  6.4 ms. reve-large is ~5.8x the parameters again, so budget ~300 ms/window on
  CPU (~25 min for one Dreyer test pass) — fine on a GPU, painful without one.
- REVE is the one backbone here designed to be used frozen ("Under linear
  probing (frozen encoder), REVE achieves state-of-the-art results"), which is
  the opposite of CBraMod's own warning that freezing causes "a very large
  performance decline". So `arm=probe` is a genuine contender for REVE and a
  diagnostic floor for CBraMod — the comparison is the interesting part.
- Norm layers differ: CBraMod is LayerNorm + GroupNorm (9,750 affine params),
  REVE is RMSNorm + LayerNorm (24,576 for base). Tent matches on any module
  whose class name contains "Norm" so neither backbone is silently half-frozen.
