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
# by ``install_solver.sh``; ``CBRAMOD_HOME`` points back at that repo so the
# vendored backbone stays importable from wherever benchopt loads this file.
_HOME = Path(os.environ.get("CBRAMOD_HOME", Path(__file__).resolve().parents[3]))
if str(_HOME) not in sys.path:
    sys.path.insert(0, str(_HOME))

from cbramod.core import (  # noqa: E402
    CBraModNet, augment_amplitude_scale, augment_stopped_band, tent_loss,
    tent_setup,
)

_WEIGHTS = Path(os.environ.get("CBRAMOD_WEIGHTS", _HOME / "weights"))
_INIT_PATHS = {
    # CBraMod's own pretrained backbone (masked EEG reconstruction).
    "pretrained": _WEIGHTS / "cbramod_pretrained.pth",
    # backbone.* of the NeuroTTT imagined-speech model (its 64ch/5-class head
    # cannot be reused here, so only the backbone was kept).
    "speech": _WEIGHTS / "cbramod_speech_neurottt.pth",
    "speech_baseline": _WEIGHTS / "cbramod_speech_baseline.pth",
}


def _resolve_init(name):
    path = _INIT_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(
            f"missing foundation weights: {path}\n"
            f"Set CBRAMOD_WEIGHTS, or check out the repo's weights/ directory."
        )
    return path


class Model:
    """Competition wrapper: ``predict(X) -> (B,)``, optionally Tent-adapted."""

    def __init__(self, net, device, tent=False, tent_lr=1e-3, tent_steps=1,
                 tent_diversity=0.0):
        self.net = net.to(device)
        self.device = device
        self.tent = tent
        self.tent_lr, self.tent_steps = tent_lr, tent_steps
        self.tent_diversity = tent_diversity
        self._tent_opt = None

    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if not self.tent:
            self.net.eval()
            with torch.inference_mode():
                return self.net(X).argmax(dim=1)

        # Stage II: adapt on this batch, then predict. State carries across
        # batches (online), which is why the test loader must stay unshuffled.
        if self._tent_opt is None:
            self._tent_opt, _ = tent_setup(self.net, lr=self.tent_lr)
        for _ in range(self.tent_steps):
            self._tent_opt.zero_grad()
            loss = tent_loss(self.net(X), diversity=self.tent_diversity)
            loss.backward()
            self._tent_opt.step()
        with torch.no_grad():
            return self.net(X).argmax(dim=1)


class Solver(CompetSolver):

    name = "CBraMod"

    parameters = {
        "arm": ["probe"],                 # probe | finetune | neurottt
        "init": ["pretrained"],           # pretrained | speech | speech_baseline
        "head": ["all_patch_reps"],       # all_patch_reps | avgpool
        "gain": [1.0],                    # input scale (CBraMod trained on uV/100)
        "tent": [False],
        "tent_lr": [1e-3],
        "tent_steps": [1],
        "tent_diversity": [0.0],
        "n_epochs": [20],
        "lr_backbone": [1e-4],
        "lr_head": [5e-4],
        "weight_decay": [5e-2],
        "label_smoothing": [0.1],
        "ssl_weight": [0.1],
        "batch_size": [64],
        "val_frac": [0.2],
        "patience": [5],
        "seed": [8888],
    }

    # ---- required -------------------------------------------------------

    def load_model(self, meta):
        net = CBraModNet(
            n_chans=meta["n_chans"], n_times=meta["n_times"],
            n_classes=meta["n_classes"], sfreq=meta["sfreq"],
            head=self.head, gain=self.gain,
            with_ssl=(self.arm == "neurottt"),
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
                print(f"[CBraMod] incompatible checkpoint {weights} — "
                      "starting from the foundation weights instead")
                net.load_backbone(_resolve_init(self.init), meta["device"])
        else:
            net.load_backbone(_resolve_init(self.init), meta["device"])
        return Model(net, meta["device"], tent=self.tent, tent_lr=self.tent_lr,
                     tent_steps=self.tent_steps,
                     tent_diversity=self.tent_diversity)

    # ---- local training only --------------------------------------------

    def fit(self, model, train_loader):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = random.Random(self.seed)

        X_all, y_all, subj = _materialise(train_loader)
        tr_idx, val_idx = _split_by_subject(subj, self.val_frac, self.seed)
        print(f"[CBraMod] arm={self.arm} init={self.init} head={self.head} "
              f"gain={self.gain} | train {len(tr_idx)} / val {len(val_idx)} "
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
def _evaluate(net, X, y, idx, device, batch=64, patched=False):
    net.eval()
    preds = []
    for i in range(0, len(idx), batch):
        xb = X[idx[i:i + batch]].to(device)
        preds.append(net(xb, patched=patched).argmax(1).cpu().numpy())
    return _bal_acc(y[idx].numpy(), np.concatenate(preds))


def _fit_probe(cfg, net, X_all, y_all, tr_idx, val_idx, device):
    """Frozen backbone: cache features once, then train the head alone."""
    for p in net.backbone.parameters():
        p.requires_grad_(False)
    net.backbone.eval()

    feats = []
    with torch.no_grad():
        for i in range(0, len(X_all), cfg.batch_size):
            xb = X_all[i:i + cfg.batch_size].to(device)
            feats.append(net.features(xb).cpu())
    F_all = torch.cat(feats)
    print(f"[CBraMod] cached backbone features {tuple(F_all.shape)} "
          f"({F_all.numel() * 4 / 1e9:.2f} GB)")

    opt = torch.optim.AdamW(net.classifier.parameters(), lr=cfg.lr_head,
                            weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best, best_state, bad = -np.inf, None, 0
    for epoch in range(cfg.n_epochs):
        net.classifier.train()
        perm = torch.randperm(len(tr_idx))
        for i in range(0, len(perm), cfg.batch_size):
            idx = tr_idx[perm[i:i + cfg.batch_size].numpy()]
            opt.zero_grad()
            loss = loss_fn(net.classifier(F_all[idx].to(device)),
                           y_all[idx].to(device))
            loss.backward()
            opt.step()

        net.classifier.eval()
        with torch.no_grad():
            pred = torch.cat([
                net.classifier(F_all[val_idx[i:i + 256]].to(device)).argmax(1).cpu()
                for i in range(0, len(val_idx), 256)
            ]).numpy()
        score = _bal_acc(y_all[val_idx].numpy(), pred)
        print(f"[CBraMod] epoch {epoch + 1:3d}/{cfg.n_epochs} "
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
                xp = net.patch(xb)
                xp_band, band_lbl = augment_stopped_band(xp, rng)
                xp_amp, amp_lbl = augment_amplitude_scale(xp, rng)
                main, band, amp = net.forward_all(xp, xp_band, xp_amp)
                main_loss = loss_fn(main, yb)
                s_loss = ssl_fn(band, band_lbl) + ssl_fn(amp, amp_lbl)
                loss = main_loss + cfg.ssl_weight * s_loss
                tot_ssl += s_loss.item()
            else:
                main_loss = loss = loss_fn(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot_main += main_loss.item()
            n_batch += 1

        score = _evaluate(net, X_all, y_all, val_idx, device, cfg.batch_size)
        msg = (f"[CBraMod] epoch {epoch + 1:3d}/{cfg.n_epochs} "
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
        print(f"[CBraMod] early stop at epoch {epoch + 1}")
        return best, best_state, bad, True
    return best, best_state, bad, False


def _restore(net, best_state, best):
    if best_state is not None:
        net.load_state_dict(best_state)
    print(f"[CBraMod] best val_bal_acc={best:.4f}  "
          "(grouped by subject — comparable to the test split)")
