"""CBraMod foundation-model arms, with the NeuroTTT recipe (track 2).

Four training arms, selected with ``arm=``:

- ``probe``     frozen backbone, head only. Backbone features are extracted
                once and cached, so this trains in seconds. The closest thing
                to "what does the pretrained representation already contain".
- ``finetune``  backbone + head end-to-end, plain cross-entropy. NeuroTTT's
                "Full Supervised Finetuning" baseline row.
- ``neurottt``  stage I: ``L = L_main + w * (L_stopband + L_amplitude)``, the
                two EEG-aware SSL heads sharing the backbone.

Stage II (Tent) is orthogonal — ``tent=True`` adapts LayerNorm affine params
by entropy minimisation over the test stream, on top of whichever arm ran.

``init=speech`` starts from the ``backbone.*`` weights of the imagined-speech
NeuroTTT checkpoint instead of the raw pretrained ones (its classifier is
64ch/3patch/5class and cannot be reused).

Examples::

    benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
        -s "CBraMod[arm=probe,finetune,neurottt]" -o "BCI-decoding[training=True]"
    benchopt run tracks/bci_decoding -d "BCI[study=dreyer2023]" \
        -s "CBraMod[arm=probe,gain=0.5,1.0,2.0]" -o "BCI-decoding[training=True]"
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmark_utils.base_solver import CompetSolver

# This solver is shipped by the bci-cbramod repo and copied into the benchmark
# by ``install_solver.sh``; ``EEGFM_HOME`` points back at that repo so the
# vendored backbone stays importable from wherever benchopt loads this file.
_HOME = Path(os.environ.get("EEGFM_HOME", Path(__file__).resolve().parents[3]))
if str(_HOME) not in sys.path:
    sys.path.insert(0, str(_HOME))

from eegfm.core import (  # noqa: E402
    FoundationNet, tent_loss, tent_setup, ttt_ssl_setup,
)

_WEIGHTS = Path(os.environ.get("EEGFM_WEIGHTS", _HOME / "weights"))
# CBraMod ships its weights in this repo (~20 MB each). REVE does not: its
# checkpoints are 0.28 GB (base) and 1.56 GB (large), so the backbone pulls
# them from the Hub on first use and caches them under HF_HOME.
_CBRAMOD_INIT = {
    # CBraMod's own pretrained backbone (masked EEG reconstruction).
    "pretrained": _WEIGHTS / "cbramod_pretrained.pth",
    # backbone.* of the NeuroTTT imagined-speech model (its 64ch/5-class head
    # cannot be reused here, so only the backbone was kept).
    "speech": _WEIGHTS / "cbramod_speech_neurottt.pth",
    "speech_baseline": _WEIGHTS / "cbramod_speech_baseline.pth",
}


def _resolve_init(backbone, name):
    if not backbone.startswith("cbramod"):
        return None                       # REVE resolves through the Hub
    path = _CBRAMOD_INIT[name]
    if not path.exists():
        raise FileNotFoundError(
            f"missing foundation weights: {path}\n"
            f"Set EEGFM_WEIGHTS, or check out the repo's weights/ directory."
        )
    return path


class Model:
    """Competition wrapper: ``predict(X) -> (B,)``, optionally Tent-adapted."""

    def __init__(self, net, device, adapt="none", tent_lr=1e-3, tent_steps=1,
                 tent_diversity=0.0, ttt_lr=1e-5, ttt_steps=1, ttt_chunk=1,
                 ttt_online=False, w_band=0.1, w_task=0.8, seed=8888):
        self.net = net.to(device)
        self.device = device
        self.adapt = adapt
        self.tent_lr, self.tent_steps = tent_lr, tent_steps
        self.tent_diversity = tent_diversity
        self.ttt_lr, self.ttt_steps = ttt_lr, ttt_steps
        self.ttt_chunk, self.ttt_online = ttt_chunk, ttt_online
        self.w_band, self.w_task = w_band, w_task
        self._opt = None
        self._snapshot = None
        self._rng = random.Random(seed)

    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if self.adapt == "none":
            self.net.eval()
            with torch.inference_mode():
                return self.net(X).argmax(dim=1)
        if self.adapt == "tent":
            return self._predict_tent(X)
        if self.adapt == "ssl":
            return self._predict_ttt_ssl(X)
        raise ValueError(f"unknown adapt={self.adapt!r}")

    def _predict_tent(self, X):
        """Stage II-b — entropy minimisation on the normalisation affines.

        State carries across batches (online), which is why the test loader
        must stay unshuffled: consecutive batches then walk through one
        recording rather than a random mixture.
        """
        if self._opt is None:
            self._opt, _ = tent_setup(self.net, lr=self.tent_lr)
        for _ in range(self.tent_steps):
            self._opt.zero_grad()
            tent_loss(self.net(X), diversity=self.tent_diversity).backward()
            self._opt.step()
        with torch.no_grad():
            return self.net(X).argmax(dim=1)

    def _predict_ttt_ssl(self, X):
        """Stage II-a — per-sample self-supervised calibration.

        The paper takes one full-parameter Adam step (lr 1e-5) on the SSL loss
        for each unlabeled test sample, predicts, then *resets* the weights
        before the next one (Table 7: batch size 1, online False). So each
        sample is personalised independently — no state leaks between trials.

        ``ttt_chunk`` relaxes the batch size of 1: it is the single biggest
        cost knob here, since the reset means the work does not amortise.
        """
        if self.net.ssl_band is None:
            raise RuntimeError(
                "adapt='ssl' needs the SSL heads — they only exist on the "
                "'neurottt' arm. Use adapt='tent' for the other arms."
            )
        if self._opt is None:
            self._opt, self._snapshot = ttt_ssl_setup(self.net, lr=self.ttt_lr)

        preds = []
        for i in range(0, len(X), self.ttt_chunk):
            xb = X[i:i + self.ttt_chunk]
            self.net.train()
            for _ in range(self.ttt_steps):
                self._opt.zero_grad()
                self.net.ssl_loss(xb, self.w_band, self.w_task,
                                  rng=self._rng).backward()
                self._opt.step()
            self.net.eval()
            with torch.no_grad():
                preds.append(self.net(xb).argmax(dim=1))
            if not self.ttt_online:
                self.net.load_state_dict(self._snapshot)
        return torch.cat(preds)


class Solver(CompetSolver):

    name = "EEGFM"

    parameters = {
        "backbone": ["cbramod"],          # cbramod | reve_base | reve_large
        "arm": ["probe"],                 # probe | finetune | neurottt
        "init": ["pretrained"],           # cbramod only: pretrained | speech | speech_baseline
        "head": ["avgpool"],              # avgpool (linear probe) | all_patch_reps
        "gain": [1.0],                    # input scale (CBraMod trained on uV/100)
        # "auto" -> zscore+clip15 for REVE (its documented preprocessing),
        # none for CBraMod (which expects roughly uV/100, i.e. ``gain``).
        "input_norm": ["auto"],
        # Stage II: none | tent (entropy, norm params, online)
        #                 | ssl  (per-sample SSL calibration, needs arm=neurottt)
        "adapt": ["none"],
        "tent_lr": [1e-3],
        "tent_steps": [1],
        "tent_diversity": [0.0],
        "ttt_lr": [1e-5],                 # paper Table 7
        "ttt_steps": [1],
        "ttt_chunk": [1],                 # paper uses batch size 1
        "ttt_online": [False],            # paper resets between samples
        # Stage I SSL: band+jigsaw is the paper's motor-imagery pair;
        # band+amp is the imagined-speech pair the reference notebook ships.
        "ssl_tasks": ["band+jigsaw"],
        "n_seg": [2],                     # jigsaw chunks (must divide n_patch)
        "w_band": [0.1],                  # paper Table 7, motor imagery
        "w_task": [0.8],
        "n_epochs": [20],
        "lr_backbone": [1e-4],
        "lr_head": [5e-4],
        "weight_decay": [5e-2],
        "label_smoothing": [0.1],
        "batch_size": [64],
        "val_frac": [0.2],
        "patience": [5],
        "seed": [8888],
    }

    # ---- required -------------------------------------------------------

    def load_model(self, meta):
        norm = self.input_norm
        if norm == "auto":
            norm = "none" if self.backbone.startswith("cbramod") else "zscore"
        net = FoundationNet(
            backbone=self.backbone,
            n_chans=meta["n_chans"], n_times=meta["n_times"],
            n_classes=meta["n_classes"], sfreq=meta["sfreq"],
            head=self.head, gain=self.gain, input_norm=norm,
            with_ssl=(self.arm == "neurottt"),
            ssl_tasks=self.ssl_tasks, task="motor_imagery", n_seg=self.n_seg,
            weights=_resolve_init(self.backbone, self.init),
            chs_info=meta.get("chs_info"), device=meta["device"],
        )
        weights = meta["submission_dir"] / "weights.pt"
        # A training run always starts from the foundation weights. Every
        # parameter configuration shares ``outputs/CBraMod`` as its default
        # submission_dir, so honouring a checkpoint here would silently
        # warm-start one arm from another arm's weights.
        training = getattr(self, "train_loader", None) is not None
        if weights.exists() and not training:
            state = torch.load(weights, map_location=meta["device"],
                               weights_only=True)
            try:
                net.load_state_dict(state)
            except RuntimeError:
                if os.environ.get("COMPET_SUBMISSION_DIR"):
                    raise      # on the platform a mismatch must never be silent
                print(f"[eegfm] incompatible checkpoint {weights} — "
                      "keeping the freshly loaded foundation weights")
        return Model(net, meta["device"], adapt=self.adapt,
                     tent_lr=self.tent_lr, tent_steps=self.tent_steps,
                     tent_diversity=self.tent_diversity,
                     ttt_lr=self.ttt_lr, ttt_steps=self.ttt_steps,
                     ttt_chunk=self.ttt_chunk, ttt_online=self.ttt_online,
                     w_band=self.w_band, w_task=self.w_task, seed=self.seed)

    # ---- local training only --------------------------------------------

    def fit(self, model, train_loader):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = random.Random(self.seed)

        X_all, y_all, subj = _materialise(train_loader)
        tr_idx, val_idx = _split_by_subject(subj, self.val_frac, self.seed)
        extra = (f" ssl={self.ssl_tasks}(w={self.w_band}/{self.w_task},"
                 f"n_seg={self.n_seg})" if self.arm == "neurottt" else "")
        print(f"[eegfm] backbone={self.backbone} arm={self.arm} init={self.init} "
              f"head={self.head} gain={self.gain}{extra} | "
              f"train {len(tr_idx)} / val {len(val_idx)} "
              f"windows over {len(np.unique(subj))} subjects")

        net = model.net
        if self.arm == "probe":
            _fit_probe(self, net, X_all, y_all, tr_idx, val_idx, model.device)
        else:
            _fit_full(self, net, X_all, y_all, tr_idx, val_idx, model.device,
                      rng, ssl=(self.arm == "neurottt"))

    def save_model(self, model, path):
        torch.save(model.net.state_dict(), path / "weights.pt")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _materialise(loader):
    """Pull the whole train split into RAM once (the loaders re-extract
    windows on every pass, which dominates epoch time)."""
    Xs, ys, ss = [], [], []
    for X, y, info in loader:
        Xs.append(X.cpu())
        ys.append(y.cpu())
        ss.append(np.asarray(info["subject_id"]).reshape(-1))
    return torch.cat(Xs), torch.cat(ys), np.concatenate(ss)


def _split_by_subject(subj, val_frac, seed):
    """Hold out whole subjects, never windows.

    The random window split in ``my_model.py`` leaks across subjects and
    inflates its validation score by ~9 points on this study. Both the warm-up
    (unseen subjects) and the sealed phase (unseen sessions) are grouped
    problems, so the model-selection split has to be grouped too.
    """
    uniq = np.unique(subj)
    rs = np.random.RandomState(seed)
    val_subj = set(rs.permutation(uniq)[:max(1, int(round(val_frac * len(uniq))))])
    is_val = np.isin(subj, list(val_subj))
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def _bal_acc(true, pred):
    return float(np.mean([
        (pred[true == k] == k).mean() for k in np.unique(true)
    ]))


@torch.no_grad()
def _evaluate(net, X, y, idx, device, batch=64, prepared=False):
    net.eval()
    preds = []
    for i in range(0, len(idx), batch):
        xb = X[idx[i:i + batch]].to(device)
        preds.append(net(xb, prepared=prepared).argmax(1).cpu().numpy())
    return _bal_acc(y[idx].numpy(), np.concatenate(preds))


def _fit_probe(cfg, net, X_all, y_all, tr_idx, val_idx, device):
    """Frozen backbone: cache features once, then train the head alone."""
    for p in net.backbone.parameters():
        p.requires_grad_(False)
    net.backbone.eval()

    # When the head pools over (channel, patch) anyway, cache the pooled
    # vector instead of every token: REVE-large emits (27, 4, 1216) per window,
    # which is 6.5 GB over the train split versus 60 MB pooled.
    pool = cfg.head == "avgpool"
    pooler = net.classifier[0] if pool else None
    head = net.classifier[1:] if pool else net.classifier
    feats = []
    with torch.no_grad():
        for i in range(0, len(X_all), cfg.batch_size):
            xb = X_all[i:i + cfg.batch_size].to(device)
            f = net.features(xb)
            feats.append((pooler(f) if pool else f).cpu())
    F_all = torch.cat(feats)
    print(f"[eegfm] cached backbone features {tuple(F_all.shape)} "
          f"({F_all.numel() * 4 / 1e9:.2f} GB, pooled={pool})")

    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr_head,
                            weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best, best_state, bad = -np.inf, None, 0
    for epoch in range(cfg.n_epochs):
        head.train()
        perm = torch.randperm(len(tr_idx))
        for i in range(0, len(perm), cfg.batch_size):
            idx = tr_idx[perm[i:i + cfg.batch_size].numpy()]
            opt.zero_grad()
            loss = loss_fn(head(F_all[idx].to(device)), y_all[idx].to(device))
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            pred = torch.cat([
                head(F_all[val_idx[i:i + 256]].to(device)).argmax(1).cpu()
                for i in range(0, len(val_idx), 256)
            ]).numpy()
        score = _bal_acc(y_all[val_idx].numpy(), pred)
        print(f"[eegfm] epoch {epoch + 1:3d}/{cfg.n_epochs} "
              f"val_bal_acc={score:.4f}")
        best, best_state, bad, stop = _track(score, best, best_state, bad,
                                             net, cfg.patience, epoch)
        if stop:
            break
    _restore(net, best_state, best)


def _fit_full(cfg, net, X_all, y_all, tr_idx, val_idx, device, rng, ssl):
    """End-to-end fine-tuning, optionally with the two NeuroTTT SSL heads."""
    groups = [
        {"params": list(net.backbone.parameters()), "lr": cfg.lr_backbone},
        {"params": list(net.classifier.parameters()), "lr": cfg.lr_head},
    ]
    if ssl:
        groups.append({"params": list(net.ssl_band.parameters())
                       + list(net.ssl_amp.parameters()), "lr": cfg.lr_head})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    ssl_fn = nn.CrossEntropyLoss()

    best, best_state, bad = -np.inf, None, 0
    for epoch in range(cfg.n_epochs):
        net.train()
        perm = torch.randperm(len(tr_idx))
        tot_main = tot_ssl = 0.0
        n_batch = 0
        for i in range(0, len(perm), cfg.batch_size):
            idx = tr_idx[perm[i:i + cfg.batch_size].numpy()]
            xb = X_all[idx].to(device)
            yb = y_all[idx].to(device)
            opt.zero_grad()
            if ssl:
                # Patch once, augment in patch space, one backbone pass for all
                # three branches (3x cheaper than three separate passes).
                xp = net.prepare(xb)
                xp_band, band_lbl, xp_task, task_lbl = net.augment(xp, rng)
                main, band, task = net.forward_all(xp, xp_band, xp_task)
                main_loss = loss_fn(main, yb)
                # Weighted per task, not one shared weight: the paper gives
                # motor imagery w_band=0.1, w_jigsaw=0.8 (Table 7).
                s_loss = (cfg.w_band * ssl_fn(band, band_lbl)
                          + cfg.w_task * ssl_fn(task, task_lbl))
                loss = main_loss + s_loss
                tot_ssl += s_loss.item()
            else:
                main_loss = loss = loss_fn(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot_main += main_loss.item()
            n_batch += 1

        score = _evaluate(net, X_all, y_all, val_idx, device, cfg.batch_size)
        msg = (f"[eegfm] epoch {epoch + 1:3d}/{cfg.n_epochs} "
               f"main={tot_main / n_batch:.4f}")
        if ssl:
            msg += f" ssl={tot_ssl / n_batch:.4f}"
        print(f"{msg} val_bal_acc={score:.4f}")
        best, best_state, bad, stop = _track(score, best, best_state, bad,
                                             net, cfg.patience, epoch)
        if stop:
            break
    _restore(net, best_state, best)


def _track(score, best, best_state, bad, net, patience, epoch):
    if score > best:
        return (score,
                {k: v.detach().clone() for k, v in net.state_dict().items()},
                0, False)
    bad += 1
    if bad >= patience:
        print(f"[eegfm] early stop at epoch {epoch + 1}")
        return best, best_state, bad, True
    return best, best_state, bad, False


def _restore(net, best_state, best):
    if best_state is not None:
        net.load_state_dict(best_state)
    print(f"[eegfm] best val_bal_acc={best:.4f}  "
          "(grouped by subject — comparable to the test split)")
