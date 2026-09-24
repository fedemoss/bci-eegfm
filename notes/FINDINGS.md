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
