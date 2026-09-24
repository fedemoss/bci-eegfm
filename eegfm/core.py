"""NeuroTTT on top of an EEG foundation backbone.

Input adaptation, the two stage-I SSL objectives, the classification heads and
both stage-II adaptation modes — all written against the backbone interface in
``backbones.py``, so CBraMod and REVE run through exactly the same code.

Augmentations are applied in **input space**, on the resampled ``(B, C, T)``
window, as the paper specifies ("the augmentations happen before the backbone").
That also keeps them correct for REVE, whose patches overlap.
"""

import itertools
import random

import torch
import torch.nn as nn

from .backbones import PATCH, TARGET_SFREQ, build_backbone

# Stopped-band SSL — the head shared by all three NeuroTTT tasks. Band
# divisions are task-specific (paper Appendix A.1, Tables 4-6).
BANDS = {
    "motor_imagery": [(3, 7), (8, 13), (13, 30), (30, 45)],      # Table 6
    "imagined_speech": [(0.5, 8), (8, 30), (30, 70), (70, 100)],  # Table 4
    "mental_stress": [(4, 8), (8, 12), (13, 20), (20, 30)],       # Table 5
}

# Amplitude-scaling SSL (imagined speech): 16 factors spanning [-2, 2].
AMP_SCALES = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
              0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


# --------------------------------------------------------------------------
# input adaptation
# --------------------------------------------------------------------------

def fft_resample(x, num):
    """FFT resampling along the last axis — ``scipy.signal.resample`` in torch.

    Differentiable and device-agnostic so it can live inside ``forward``: the
    competition platform hands raw windows with no preprocessing hook.
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


def prepare_input(X, sfreq, gain=1.0, input_norm="none", clip=15.0):
    """``(B, C, T)`` at ``sfreq`` -> whole 1 s patches at 200 Hz, scaled.

    ``input_norm="zscore"`` applies REVE's documented preprocessing: per-channel
    z-score then clipping at ``clip`` standard deviations. CBraMod instead
    expects roughly uV/100, which is what ``gain`` is for.
    """
    n_out = int(round(X.shape[-1] * TARGET_SFREQ / sfreq))
    X = fft_resample(X, max(PATCH, (n_out // PATCH) * PATCH))
    if input_norm == "zscore":
        mu = X.mean(dim=-1, keepdim=True)
        sd = X.std(dim=-1, keepdim=True).clamp_min(1e-6)
        X = ((X - mu) / sd).clamp(-clip, clip)
    return X * gain


# --------------------------------------------------------------------------
# stage-I SSL augmentations — input space, (B, C, T) at 200 Hz
# --------------------------------------------------------------------------

def augment_stopped_band(x, bands, rng=random):
    """Zero one random frequency band per sample; return (x, band_label)."""
    B = x.shape[0]
    idx = torch.tensor([rng.randrange(len(bands)) for _ in range(B)],
                       device=x.device)
    Xf = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / TARGET_SFREQ).to(x.device)
    lo = torch.tensor([bands[i][0] for i in idx.tolist()], device=x.device)
    hi = torch.tensor([bands[i][1] for i in idx.tolist()], device=x.device)
    mask = (freqs[None] >= lo[:, None]) & (freqs[None] < hi[:, None])
    Xf = Xf.masked_fill(mask[:, None, :], 0)
    return torch.fft.irfft(Xf, n=x.shape[-1], dim=-1), idx


def jigsaw_permutations(n_seg):
    """Label space of the jigsaw head: every ordering of ``n_seg`` chunks."""
    return list(itertools.permutations(range(n_seg)))


def augment_temporal_jigsaw(x, n_seg=2, rng=random):
    """Motor-imagery SSL: shuffle consecutive time chunks, predict the order.

    Paper A.1: split the trial into two or three equal consecutive segments,
    shuffle them, and have the model recover the order. With two segments this
    is chronological-vs-reversed. The pretext forces the backbone to encode
    temporal progression — which for MI is the ERD/ERS time course after the
    cue, exactly the discriminative structure.
    """
    B, C, T = x.shape
    if T % n_seg:
        raise ValueError(f"n_seg={n_seg} must divide the window length {T}")
    perms = jigsaw_permutations(n_seg)
    idx = torch.tensor([rng.randrange(len(perms)) for _ in range(B)],
                       device=x.device)
    chunks = x.view(B, C, n_seg, T // n_seg)
    out = torch.empty_like(chunks)
    for i, label in enumerate(idx.tolist()):
        out[i] = chunks[i, :, list(perms[label])]
    return out.reshape(B, C, T), idx


def augment_amplitude_scale(x, rng=random):
    """Imagined-speech SSL: scale the window, predict which factor was used."""
    B = x.shape[0]
    idx = torch.tensor([rng.randrange(len(AMP_SCALES)) for _ in range(B)],
                       device=x.device)
    scale = torch.tensor(AMP_SCALES, device=x.device, dtype=x.dtype)[idx]
    return x * scale.view(B, 1, 1), idx


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------

class MeanPool(nn.Module):
    """``(B, C, S, D) -> (B, D)`` — mean over channels and patches."""

    def forward(self, x):
        return x.mean(dim=(1, 2))


def make_head(kind, n_chans, n_patch, embed_dim, n_classes, dropout=0.1):
    """Classifier over backbone features ``(B, C, S, D)``."""
    if kind == "avgpool":
        # A true linear probe: embed_dim * n_classes parameters. The right
        # choice when the train split is ~12k windows, and the setting REVE
        # was designed for.
        return nn.Sequential(MeanPool(), nn.Linear(embed_dim, n_classes))
    if kind == "all_patch_reps":
        # CBraMod's own BCIC-IV-2a head: flatten every token.
        flat = n_chans * n_patch * embed_dim
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(flat, n_patch * 200), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(n_patch * 200, 200), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(200, n_classes),
        )
    raise ValueError(f"unknown head {kind!r}")


class SSLHead(nn.Module):
    """NeuroTTT auxiliary head on pooled backbone features."""

    def __init__(self, embed_dim, n_class, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_class),
        )

    def forward(self, feat):
        return self.net(feat.mean(dim=(1, 2)))


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class FoundationNet(nn.Module):
    """Input adaptation + a foundation backbone + task head (+ SSL heads)."""

    def __init__(self, backbone, n_chans, n_times, n_classes, sfreq,
                 head="avgpool", gain=1.0, input_norm="none", dropout=0.1,
                 with_ssl=False, ssl_tasks="band+jigsaw", task="motor_imagery",
                 n_seg=2, weights=None, chs_info=None, device="cpu"):
        super().__init__()
        self.sfreq, self.gain, self.input_norm = float(sfreq), float(gain), input_norm
        self.n_times_200 = max(PATCH, (int(round(n_times * TARGET_SFREQ / sfreq))
                                       // PATCH) * PATCH)
        self.backbone = build_backbone(
            backbone, weights=weights, n_chans=n_chans,
            n_times=self.n_times_200, chs_info=chs_info, device=device)
        D = self.backbone.embed_dim
        self.n_patch = self.backbone.n_patches(self.n_times_200)
        self.classifier = make_head(head, n_chans, self.n_patch, D, n_classes,
                                    dropout)

        self.ssl_tasks, self.bands, self.n_seg = ssl_tasks, BANDS[task], n_seg
        if with_ssl:
            self.ssl_band = SSLHead(D, len(self.bands), dropout=dropout)
            n_task = (len(jigsaw_permutations(n_seg))
                      if ssl_tasks.endswith("jigsaw") else len(AMP_SCALES))
            self.ssl_task = SSLHead(D, n_task, dropout=dropout)
        else:
            self.ssl_band = self.ssl_task = None

    def prepare(self, X):
        return prepare_input(X, self.sfreq, self.gain, self.input_norm)

    def features(self, X, prepared=False):
        return self.backbone.encode(X if prepared else self.prepare(X))

    def forward(self, X, prepared=False):
        return self.classifier(self.features(X, prepared=prepared))

    def augment(self, x, rng=random):
        """The two SSL views of a prepared window, with their labels."""
        x_band, y_band = augment_stopped_band(x, self.bands, rng)
        if self.ssl_tasks.endswith("jigsaw"):
            x_task, y_task = augment_temporal_jigsaw(x, self.n_seg, rng)
        else:
            x_task, y_task = augment_amplitude_scale(x, rng)
        return x_band, y_band, x_task, y_task

    def forward_all(self, x, x_band, x_task):
        """One backbone pass over the three branches (already prepared)."""
        B = x.shape[0]
        feats = self.backbone.encode(torch.cat([x, x_band, x_task], dim=0))
        return (self.classifier(feats[:B]),
                self.ssl_band(feats[B:2 * B]),
                self.ssl_task(feats[2 * B:]))

    def ssl_loss(self, X, w_band, w_task, rng=random, prepared=False):
        """Label-free SSL objective.

        Stage I adds this to the supervised loss; stage II-a descends it alone
        on each unlabeled test sample. One implementation for both, so the
        objective optimised at test time is provably the one trained on — the
        premise of test-time *training*.
        """
        x = X if prepared else self.prepare(X)
        x_band, y_band, x_task, y_task = self.augment(x, rng)
        feats = self.backbone.encode(torch.cat([x_band, x_task], dim=0))
        B = x.shape[0]
        ce = nn.functional.cross_entropy
        return (w_band * ce(self.ssl_band(feats[:B]), y_band)
                + w_task * ce(self.ssl_task(feats[B:]), y_task))


# --------------------------------------------------------------------------
# stage II — adaptation
# --------------------------------------------------------------------------

def norm_affine_params(net):
    """Affine parameters of every normalisation layer, whatever its class.

    The paper describes Tent as updating BatchNorm statistics. CBraMod is
    LayerNorm throughout and REVE mixes LayerNorm with RMSNorm, so matching on
    ``nn.LayerNorm`` alone would silently leave most of REVE's norms frozen.
    """
    mods, params = [], []
    for m in net.modules():
        if "Norm" not in type(m).__name__:
            continue
        for attr in ("weight", "bias"):
            p = getattr(m, attr, None)
            if isinstance(p, nn.Parameter):
                params.append(p)
        if params and m not in mods:
            mods.append(m)
    return mods, params


def tent_setup(net, lr=1e-3):
    """Stage II-b: freeze everything but the normalisation affines."""
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    mods, params = norm_affine_params(net)
    if not params:
        raise RuntimeError("no affine normalisation layer — Tent has nothing to adapt")
    for m in mods:
        m.train()
    for p in params:
        p.requires_grad_(True)
    return torch.optim.SGD(params, lr=lr), params


def ttt_ssl_setup(net, lr=1e-5):
    """Stage II-a: full-parameter Adam over the SSL objective (paper Table 7).

    Returns the optimizer and a snapshot of the original weights — the paper
    runs ``online=False``, resetting to this snapshot before the next test
    sample so each one is calibrated independently.
    """
    for p in net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    snapshot = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return opt, snapshot


def tent_loss(logits, diversity=0.0):
    """Entropy minimisation, optionally regularised against class collapse.

    Plain Tent can drive every window to one class; *balanced* accuracy
    punishes that far harder than plain accuracy would, so ``diversity`` adds
    back the entropy of the marginal prediction.
    """
    logp = torch.log_softmax(logits, dim=1)
    p = logp.exp()
    ent = -(p * logp).sum(1).mean()
    if diversity:
        p_bar = p.mean(0)
        ent = ent - diversity * -(p_bar * torch.log(p_bar + 1e-8)).sum()
    return ent
