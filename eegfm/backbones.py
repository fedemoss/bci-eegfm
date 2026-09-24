"""EEG foundation-model backbones behind one interface.

Every backbone takes a raw window ``(B, C, T)`` already resampled to 200 Hz —
the rate both CBraMod and REVE were pretrained at — and returns token features
``(B, C, S, D)``: one D-dimensional vector per (channel, patch).

That shared shape is what lets the arms, the SSL heads and the adaptation
methods be written once.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

PATCH = 200          # 1 s at 200 Hz, fixed by both models' pretraining
TARGET_SFREQ = 200.0

# REVE checkpoints on the Hub. Fetched by ``from_pretrained`` on first use and
# cached under HF_HOME; nothing is vendored (reve-large alone is 1.56 GB).
REVE_REPOS = {
    "reve_base": "brain-bzh/reve-base",     # 69M params, embed 512
    "reve_large": "brain-bzh/reve-large",   # 400M params, embed 1216
}


class Backbone(nn.Module):
    """Interface: ``encode((B, C, T) @ 200 Hz) -> (B, C, S, D)``."""

    embed_dim: int

    def encode(self, x):
        raise NotImplementedError

    def n_patches(self, n_times):
        raise NotImplementedError


class CBraModBackbone(Backbone):
    """CBraMod — criss-cross transformer over non-overlapping 1 s patches."""

    def __init__(self, weights=None, device="cpu"):
        super().__init__()
        from .cbramod.cbramod import CBraMod
        self.net = CBraMod(in_dim=200, out_dim=200, d_model=200,
                           dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        self.embed_dim = 200
        if weights is not None:
            self._load(weights, device)
        # The pretraining reconstruction head is not part of the representation.
        self.net.proj_out = nn.Identity()

    def _load(self, path, device):
        sd = torch.load(Path(path), map_location=device, weights_only=True)
        if any(k.startswith("backbone.") for k in sd):
            # A fine-tuned NeuroTTT checkpoint — keep only the backbone.
            sd = {k[len("backbone."):]: v for k, v in sd.items()
                  if k.startswith("backbone.")}
        missing, _ = self.net.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("proj_out")]
        if missing:
            raise RuntimeError(f"backbone weights missing keys: {missing[:6]}")

    def n_patches(self, n_times):
        return max(1, n_times // PATCH)

    def encode(self, x):
        B, C, T = x.shape
        n_patch = self.n_patches(T)
        return self.net(x[..., :n_patch * PATCH].reshape(B, C, n_patch, PATCH))


class ReveBackbone(Backbone):
    """REVE — 4D (x, y, z, t) positional encoding over overlapping patches.

    Unlike CBraMod's convolutional positional encoding, REVE's Fourier 4D
    scheme takes real electrode coordinates, so it transfers to an unseen
    montage without retraining — which is why it is the one backbone here that
    is actually designed to be used frozen.
    """

    def __init__(self, repo, n_chans, n_times, chs_info, device="cpu"):
        super().__init__()
        from braindecode.models import REVE
        self.net = REVE.from_pretrained(
            REVE_REPOS.get(repo, repo), n_outputs=2, n_chans=n_chans,
            n_times=n_times, sfreq=TARGET_SFREQ, chs_info=chs_info,
        )
        self.embed_dim = self.net.embed_dim
        # The classification head is rebuilt per arm; drop REVE's own.
        self.net.final_layer = nn.Identity()

    def n_patches(self, n_times):
        stride = self.net.patch_size - self.net.patch_overlap
        return max(1, (n_times - self.net.patch_size) // stride + 1)

    def encode(self, x):
        return self.net(x, return_features=True)["features"]


def build_backbone(name, *, weights=None, n_chans=None, n_times=None,
                   chs_info=None, device="cpu"):
    if name == "cbramod":
        return CBraModBackbone(weights=weights, device=device)
    if name in REVE_REPOS or name.startswith("brain-bzh/"):
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            print(f"[eegfm] HF_HUB_OFFLINE=1 — {name} must already be in "
                  f"HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
        return ReveBackbone(name, n_chans, n_times, chs_info, device)
    raise ValueError(f"unknown backbone {name!r}")
