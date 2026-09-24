"""CBraMod + NeuroTTT building blocks, shared by the benchopt solver.

Dreyer windows arrive as ``(B, 27, 480)`` @ 120 Hz. CBraMod expects
``(B, C, S, 200)`` with one patch = 1 s @ 200 Hz, so a 4 s window is
resampled 480 -> 800 and viewed as 4 patches. This mirrors CBraMod's own
``preprocessing_bciciv2a.py`` (``resample(sample, 800)`` then
``reshape(22, 4, 200)``).
"""

import math
import random
from pathlib import Path

import torch
import torch.nn as nn

from .models.cbramod import CBraMod

PATCH = 200          # samples per patch = 1 s @ 200 Hz (fixed by pretraining)
TARGET_SFREQ = 200.0

# Stopped-band SSL: which band was zeroed out (5-way).
BAND_EDGES = [(0, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
# Amplitude-scaling SSL: which factor was applied (4-way).
AMP_SCALES = [0.5, 0.75, 1.25, 1.5]


# --------------------------------------------------------------------------
# input adaptation
# --------------------------------------------------------------------------

def fft_resample(x, num):
    """FFT resampling along the last axis — ``scipy.signal.resample`` in torch.

    Kept differentiable and device-agnostic so it can live inside ``forward``
    (the platform hands raw windows with no preprocessing hook).
    """
    n = x.shape[-1]
    if n == num:
        return x
    X = torch.fft.rfft(x, dim=-1)
    n_out = num // 2 + 1
    keep = min(X.shape[-1], n_out)
    Y = X.new_zeros(*X.shape[:-1], n_out)
    Y[..., :keep] = X[..., :keep]
    # scipy halves the Nyquist bin when up-sampling from an even-length input.
    if n % 2 == 0 and num > n:
        Y[..., n // 2] *= 0.5
    return torch.fft.irfft(Y, n=num, dim=-1) * (num / n)


def to_patches(X, sfreq, gain=1.0):
    """``(B, C, T)`` at ``sfreq`` -> ``(B, C, S, 200)`` at 200 Hz, scaled."""
    n_out = int(round(X.shape[-1] * TARGET_SFREQ / sfreq))
    n_patch = max(1, n_out // PATCH)
    X = fft_resample(X, n_patch * PATCH)
    B, C, _ = X.shape
    return X.reshape(B, C, n_patch, PATCH) * gain


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def make_head(kind, n_chans, n_patch, n_classes, dropout=0.1):
    """Classifier over backbone features ``(B, C, S, 200)``."""
    if kind == "avgpool":
        # 402 params — the sane choice when the train split is ~12k windows.
        return nn.Sequential(
            _Rearrange4to3(),                       # (B, C, S, D) -> (B, D)
            nn.Linear(200, n_classes),
        )
    if kind == "all_patch_reps":
        # CBraMod's own BCIC-IV-2a head: flatten everything (17.4M params here).
        flat = n_chans * n_patch * 200
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(flat, n_patch * 200), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(n_patch * 200, 200), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(200, n_classes),
        )
    raise ValueError(f"unknown head {kind!r}")


class _Rearrange4to3(nn.Module):
    """Mean over channels and patches: ``(B, C, S, D) -> (B, D)``."""

    def forward(self, x):
        return x.mean(dim=(1, 2))


class SSLHead(nn.Module):
    """NeuroTTT auxiliary head on pooled backbone features."""

    def __init__(self, n_class, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(200, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_class),
        )

    def forward(self, feat):
        return self.net(feat.mean(dim=(1, 2)))


class CBraModNet(nn.Module):
    """Patching + CBraMod backbone + task head (+ optional SSL heads)."""

    def __init__(self, n_chans, n_times, n_classes, sfreq,
                 head="all_patch_reps", gain=1.0, dropout=0.1, with_ssl=False):
        super().__init__()
        self.sfreq, self.gain = float(sfreq), float(gain)
        self.n_patch = max(1, int(round(n_times * TARGET_SFREQ / sfreq)) // PATCH)
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200,
                                dim_feedforward=800, seq_len=30,
                                n_layer=12, nhead=8)
        self.backbone.proj_out = nn.Identity()
        self.classifier = make_head(head, n_chans, self.n_patch, n_classes,
                                    dropout)
        self.ssl_band = SSLHead(len(BAND_EDGES), dropout=dropout) if with_ssl else None
        self.ssl_amp = SSLHead(len(AMP_SCALES), dropout=dropout) if with_ssl else None

    def patch(self, X):
        return to_patches(X, self.sfreq, self.gain)

    def features(self, X, patched=False):
        return self.backbone(X if patched else self.patch(X))

    def forward(self, X, patched=False):
        return self.classifier(self.features(X, patched=patched))

    def forward_all(self, xp, xp_band, xp_amp):
        """One backbone pass over the three branches (already patched)."""
        B = xp.shape[0]
        feats = self.backbone(torch.cat([xp, xp_band, xp_amp], dim=0))
        return (self.classifier(feats[:B]),
                self.ssl_band(feats[B:2 * B]),
                self.ssl_amp(feats[2 * B:]))

    def load_backbone(self, path, device="cpu"):
        """Load pretrained CBraMod weights, or the ``backbone.*`` subset of a
        fine-tuned NeuroTTT checkpoint."""
        sd = torch.load(Path(path), map_location=device, weights_only=True)
        if any(k.startswith("backbone.") for k in sd):
            sd = {k[len("backbone."):]: v for k, v in sd.items()
                  if k.startswith("backbone.")}
        # ``proj_out`` was replaced by Identity, so its keys are now unexpected.
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("proj_out")]
        if missing:
            raise RuntimeError(f"backbone weights missing keys: {missing[:6]}")
        return self


# --------------------------------------------------------------------------
# NeuroTTT stage I — SSL augmentations (applied in patch space, 200 Hz)
# --------------------------------------------------------------------------

def augment_stopped_band(xp, rng=random):
    """Zero one random frequency band per sample; return (x, band_label)."""
    B = xp.shape[0]
    idx = torch.tensor([rng.randrange(len(BAND_EDGES)) for _ in range(B)],
                       device=xp.device)
    Xf = torch.fft.rfft(xp, dim=-1)
    freqs = torch.fft.rfftfreq(xp.shape[-1], d=1.0 / TARGET_SFREQ).to(xp.device)
    # (B, n_freq) mask: True where this sample's chosen band sits.
    lo = torch.tensor([BAND_EDGES[i][0] for i in idx.tolist()], device=xp.device)
    hi = torch.tensor([BAND_EDGES[i][1] for i in idx.tolist()], device=xp.device)
    mask = (freqs[None] >= lo[:, None]) & (freqs[None] < hi[:, None])
    Xf = Xf.masked_fill(mask[:, None, None, :], 0)
    return torch.fft.irfft(Xf, n=xp.shape[-1], dim=-1), idx


def augment_amplitude_scale(xp, rng=random):
    """Scale each sample by a random factor; return (x, scale_label)."""
    B = xp.shape[0]
    idx = torch.tensor([rng.randrange(len(AMP_SCALES)) for _ in range(B)],
                       device=xp.device)
    scale = torch.tensor(AMP_SCALES, device=xp.device)[idx]
    return xp * scale.view(B, 1, 1, 1), idx


# --------------------------------------------------------------------------
# NeuroTTT stage II — Tent
# --------------------------------------------------------------------------

def tent_setup(net, lr=1e-3):
    """Freeze everything but the LayerNorm affine parameters (Tent)."""
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    params = []
    for m in net.modules():
        if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            m.train()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            params += [m.weight, m.bias]
    if not params:
        raise RuntimeError("no affine LayerNorm found — Tent has nothing to adapt")
    return torch.optim.SGD(params, lr=lr), params


def tent_loss(logits, diversity=0.0):
    """Entropy minimisation, optionally regularised against class collapse.

    Plain Tent can drive every window to one class, which balanced accuracy
    punishes far harder than plain accuracy — ``diversity`` adds back the
    (negated) entropy of the marginal prediction to counteract that.
    """
    logp = torch.log_softmax(logits, dim=1)
    p = logp.exp()
    ent = -(p * logp).sum(1).mean()
    if diversity:
        p_bar = p.mean(0)
        ent_bar = -(p_bar * torch.log(p_bar + 1e-8)).sum()
        ent = ent - diversity * ent_bar
    return ent
