# bci-cbramod

CBraMod + NeuroTTT arms for **Neural Interfaces 2026, Track 2 (BCI decoding)**,
evaluated on the warm-up study **Dreyer2023Large** (27 ch, 2-class motor
imagery, test = Part B subjects 61–81).

Everything runs through the competition's own benchopt objective, so the numbers
sit on the same scale as the shipped baselines:

| reference (same split, same metric) | test balanced accuracy |
|---|---|
| `EEGNet` — competition baseline, 2,420 params | 0.7651 |
| `MyModel` — EEGNet + per-window z-score | 0.7794 |
| chance | 0.50 |

---

## Quick start on the server

```bash
git clone <this-repo> ~/bci-cbramod && cd ~/bci-cbramod

# 1. login node — builds the conda env, clones the benchmark, installs the solver
nvidia-smi                                  # check the driver, pick the channel
TORCH_CHANNEL=https://download.pytorch.org/whl/cu126 bash setup.sh

# 2. login node — stages ~19 GB of Dreyer2023 (compute nodes have no network)
source env.sh && bash prepare_data.sh

# 3. 5-minute plumbing check on a small study (scores are meaningless, paths aren't)
bash run/smoke.sh

# 4. the real thing
sbatch run/cbramod.sbatch                   # or: bash run/matrix.sh
python run/summarize.py
```

Edit `--partition` / `--account` in `run/cbramod.sbatch` before submitting.

**Put the data on scratch.** `WORK` defaults to `./work` inside the repo; set it
to a scratch filesystem instead, and keep it exported everywhere:

```bash
export WORK=/scratch/$USER/bci        # before sourcing env.sh
```

After every `git pull`, re-run `bash install_solver.sh` — benchopt discovers
solvers by scanning the benchmark's `solvers/` folder, so the file has to
physically live there.

---

## What gets run

Three training arms × two backbone initialisations, each scored with and
without test-time adaptation.

| `arm=` | what trains | cost/epoch (laptop CPU; GPU is 1–2 orders faster) |
|---|---|---|
| `probe` | frozen backbone, head only — features cached once | seconds |
| `finetune` | backbone + head, plain CE — the paper's *Full Supervised Finetuning* | ~6.5 min |
| `neurottt` | stage I: `L = L_main + Σⱼ wⱼ·L_ssl⁽ʲ⁾` | ~20 min |

Note the paper keeps the backbone **fully trainable in every setting** (§4.1) and
reports LoRA — which freezes it — as the *worst* strategy on motor imagery
(0.4164 vs 0.5028 for a full fine-tune). `probe` freezes it completely, so treat
it as a diagnostic floor ("what is already in the representation"), not a
contender.

### Stage I SSL: the pair is task-specific

Appendix A.1. Stopped-Band Prediction is shared by all three NeuroTTT tasks;
the second head is chosen per task, and **motor imagery gets Temporal Jigsaw**:

| `ssl_tasks=` | heads | for |
|---|---|---|
| `band+jigsaw` *(default)* | stopped-band (4-way) + temporal jigsaw (`n_seg!`-way) | motor imagery — the paper's MI pair |
| `band+amp` | stopped-band + amplitude scaling (16-way) | imagined speech — what the reference notebook ships |

MI band divisions are the paper's Table 6 (θ 3–7, μ 8–13, β 13–30, low-γ 30–45),
and the loss weights default to Table 7's motor-imagery row: `w_band=0.1`,
`w_task=0.8` — the jigsaw term dominates.

### Stage II: two adaptation modes, not one

| `adapt=` | what it does | cost |
|---|---|---|
| `none` | frozen weights at test time | — |
| `tent` | entropy minimisation on the normalisation affines, online across batches | ~1 extra fwd+bwd per batch |
| `ssl` | **per-sample calibration**: one full-parameter Adam step (lr 1e-5) on the SSL loss for each unlabeled test sample, predict, then **reset** before the next one | ~2 fwd + 1 bwd *per sample* |

`adapt=ssl` is the paper's "TTT with SSL" and needs the SSL heads, so it only
runs on `arm=neurottt`. It follows Table 7 (steps 1, batch size 1, lr 1e-5,
`online=False`); `ttt_chunk` relaxes the batch size of 1 and is the dominant
cost knob, since the reset means the work does not amortise across a batch.
`ttt_online=True` carries state forward instead of resetting.

CBraMod has **no BatchNorm** — the paper describes Tent as updating BN
statistics, so here it adapts the 24 LayerNorm affines (9,600 of 4.99M
parameters, 0.19%).

| `init=` | backbone weights |
|---|---|
| `pretrained` | CBraMod's own pretrained backbone (masked EEG reconstruction) |
| `speech` | `backbone.*` of the NeuroTTT imagined-speech model — tests cross-task transfer |

`tent_diversity=1.0` adds a marginal-entropy term — plain Tent can collapse to a
single class, which *balanced* accuracy punishes far harder than plain accuracy
would, so the matrix reports both.

`run/matrix.sh` trains each arm into its own `COMPET_SUBMISSION_DIR`, then
re-scores it inference-only under each adaptation mode. The unadapted number
comes from the training run itself.

### Stage 0: the input gain

CBraMod was pretrained on µV/100; Dreyer arrives already RobustScaler'd, so
whether the scales match is an empirical question:

```bash
GAIN_SWEEP=1 ARMS= bash run/matrix.sh      # probe arm × gain × head, cheap
GAIN=2.0 bash run/matrix.sh                # then run the matrix at the winner
```

Pick the gain on the **subject-grouped validation** score the solver prints, not
on the test column.

### Knobs

```bash
ARMS="probe finetune neurottt"   INITS="pretrained speech"
GAIN=1.0  EPOCHS=20  TENT_DIV=1.0
DATASET="BCI[study=dreyer2023,num_workers=4]"
```

All are environment variables read by `run/matrix.sh`.

---

## Layout

```
env.sh              paths + conda activation — source this first
setup.sh            one-time: conda env, clone the benchmark, install the solver
prepare_data.sh     stage Dreyer2023 (~19 GB, login node)
install_solver.sh   copy solvers/cbramod.py into the benchmark (re-run after git pull)
cbramod/
  models/           vendored CBraMod backbone (from wjq-learning/CBraMod)
  core.py           patching, SSL augmentations, Tent
  smoke_test.py     loads the weights + benchmarks this machine
solvers/cbramod.py  the benchopt solver — all arms live here
weights/            three backbone checkpoints, ~20 MB each
run/
  smoke.sh          5-minute plumbing check
  matrix.sh         the experiment matrix
  cbramod.sbatch    slurm wrapper
  summarize.py      joins grouped-val (from logs) with test (from parquet)
notes/FINDINGS.md   what was measured before this repo existed — read this
work/               gitignored: benchmark clone, data, caches, results, logs
```

---

## Three things that will bite you

**The shipped NeuroTTT checkpoints are imagined speech, not motor imagery.**
`neurottt.pth` / `baseline.pth` are BCIC2020-III models: 64 ch × 3 patches → 5
classes. Their heads cannot be reused for Dreyer, so `weights/cbramod_speech_*.pth`
keep only the `backbone.*` keys. And there is no pretrained NeuroTTT to evaluate —
Stage I is a supervised+SSL fine-tuning stage by construction. See
[notes/FINDINGS.md](notes/FINDINGS.md).

**Validation is grouped by subject.** A random window split leaks across
subjects and inflated the earlier scaffold's validation score by ~9 points on
this study. Warm-up is scored on unseen subjects, the sealed phase on unseen
sessions — both grouped problems.

**A training run ignores any existing `weights.pt`.** Every benchopt parameter
configuration shares `outputs/<solver-name>/` as its submission dir, so
honouring a checkpoint there would silently warm-start one arm from another
arm's weights.

## Not submission-ready

`solvers/cbramod.py` imports from the `cbramod/` package. The competition
contract allows **one file only**, so uploading to Codabench needs the backbone
inlined into a single `submission.py` (~350 lines) plus ~20 MB of weights in the
ZIP. Fine for benchmarking; do it before you upload.

## Credits and licensing

The vendored backbone (`cbramod/models/`) and the checkpoints in `weights/` come
from [wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod), MIT
licensed, (c) 2025 Jiquan Wang — see `cbramod/models/{LICENSE,NOTICE}`. The only
change to that code is making one import package-relative.
NeuroTTT recipe: [arXiv:2509.26301](https://arxiv.org/abs/2509.26301).
Benchmark: [neural-interfaces26/2026-competition](https://github.com/neural-interfaces26/2026-competition).
