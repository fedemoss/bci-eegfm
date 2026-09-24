"""CBraMod smoke test + CPU feasibility benchmark for the Dreyer2023 warm-up task.

Dreyer windows arrive as (B, 27, 480) @ 120 Hz. CBraMod wants (B, C, S, 200)
with 200 = 1 s @ 200 Hz, so a window becomes 800 samples -> 4 patches.
"""
import sys, time, resource
from pathlib import Path

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbramod.models.cbramod import CBraMod

import os
_HOME = Path(os.environ.get("CBRAMOD_HOME", Path(__file__).resolve().parents[1]))
PRETRAINED = Path(os.environ.get("CBRAMOD_WEIGHTS", _HOME / "weights")) / "cbramod_pretrained.pth"

N_CHANS, N_PATCH, PATCH, N_CLASSES = 27, 4, 200, 2
torch.set_num_threads(4)


def build(n_chans=N_CHANS, n_classes=N_CLASSES, load=True):
    backbone = CBraMod(in_dim=200, out_dim=200, d_model=200,
                       dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    if load:
        sd = torch.load(PRETRAINED, map_location="cpu", weights_only=True)
        missing, unexpected = backbone.load_state_dict(sd, strict=True)
    backbone.proj_out = nn.Identity()
    head = nn.Sequential(
        Rearrange("b c s d -> b (c s d)"),
        nn.Linear(n_chans * N_PATCH * 200, N_PATCH * 200), nn.ELU(), nn.Dropout(0.1),
        nn.Linear(N_PATCH * 200, 200), nn.ELU(), nn.Dropout(0.1),
        nn.Linear(200, n_classes),
    )
    return nn.Sequential(backbone, head)


def timeit(fn, n=3):
    fn()                                   # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


if __name__ == "__main__":
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | cuda {torch.cuda.is_available()}")

    model = build()
    bb, head = model[0], model[1]
    n_bb = sum(p.numel() for p in bb.parameters())
    n_hd = sum(p.numel() for p in head.parameters())
    print(f"\nbackbone   {n_bb/1e6:7.2f}M params   (pretrained weights loaded OK)")
    print(f"head       {n_hd/1e6:7.2f}M params   ({N_CHANS}x{N_PATCH}x200 -> {N_CLASSES})")
    print(f"total      {(n_bb+n_hd)/1e6:7.2f}M params")

    for bs in (16, 32, 64):
        x = torch.randn(bs, N_CHANS, N_PATCH, PATCH)
        model.eval()
        with torch.inference_mode():
            t_fwd = timeit(lambda: model(x))
        model.train()

        def step():
            out = model(x)
            loss = nn.functional.cross_entropy(out, torch.randint(0, N_CLASSES, (bs,)))
            loss.backward()
            model.zero_grad(set_to_none=True)
        t_bwd = timeit(step, n=2)
        print(f"\nbatch {bs:3d}  forward {t_fwd*1e3:8.1f} ms  ({t_fwd/bs*1e3:5.2f} ms/window)"
              f"   fwd+bwd {t_bwd*1e3:8.1f} ms  ({t_bwd/bs*1e3:5.2f} ms/window)")

    with torch.inference_mode():
        model.eval()
        out = model(torch.randn(4, N_CHANS, N_PATCH, PATCH))
    print(f"\nsmoke forward OK -> logits {tuple(out.shape)}")
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
