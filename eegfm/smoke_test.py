"""Load each backbone and benchmark this machine on Dreyer-shaped input.

    python -m eegfm.smoke_test               # cbramod only (no download)
    python -m eegfm.smoke_test reve_base     # also pulls 0.28 GB from the Hub
    python -m eegfm.smoke_test reve_large    # 1.56 GB
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegfm.core import FoundationNet  # noqa: E402

# Dreyer2023 warm-up shape.
N_CHANS, N_TIMES, SFREQ, N_CLASSES = 27, 480, 120.0, 2
CH_NAMES = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4', 'C6',
            'F4', 'FC2', 'FC4', 'FC6', 'CP2', 'CP4', 'CP6', 'P4', 'F3', 'FC1',
            'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3']
_HOME = Path(os.environ.get("EEGFM_HOME", Path(__file__).resolve().parents[1]))
_WEIGHTS = Path(os.environ.get("EEGFM_WEIGHTS", _HOME / "weights"))

torch.set_num_threads(4)


def build(backbone, head="avgpool", with_ssl=True):
    return FoundationNet(
        backbone=backbone, n_chans=N_CHANS, n_times=N_TIMES,
        n_classes=N_CLASSES, sfreq=SFREQ, head=head, with_ssl=with_ssl,
        input_norm="none" if backbone.startswith("cbramod") else "zscore",
        weights=_WEIGHTS / "cbramod_pretrained.pth"
        if backbone.startswith("cbramod") else None,
        chs_info=[{"ch_name": c} for c in CH_NAMES],
    )


def timeit(fn, n=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main(names):
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"cuda {torch.cuda.is_available()}")
    for name in names:
        t0 = time.perf_counter()
        net = build(name)
        load = time.perf_counter() - t0
        n_bb = sum(p.numel() for p in net.backbone.parameters())
        n_hd = sum(p.numel() for p in net.classifier.parameters())
        x = torch.randn(8, N_CHANS, N_TIMES)
        net.eval()
        with torch.no_grad():
            feats = net.features(x)
            out = net(x)
            t_fwd = timeit(lambda: net(x))
        net.train()

        def step():
            nn.functional.cross_entropy(
                net(x), torch.randint(0, N_CLASSES, (8,))).backward()
            net.zero_grad(set_to_none=True)
        t_bwd = timeit(step, n=2)

        print(f"\n{name}  (loaded in {load:.1f}s)")
        print(f"  backbone {n_bb / 1e6:7.2f}M params, embed_dim {net.backbone.embed_dim}, "
              f"{net.n_patch} patches")
        print(f"  head     {n_hd / 1e6:7.2f}M params -> logits {tuple(out.shape)}")
        print(f"  features {tuple(feats.shape)}")
        print(f"  forward  {t_fwd / 8 * 1000:7.1f} ms/window   "
              f"fwd+bwd {t_bwd / 8 * 1000:7.1f} ms/window")
        print(f"  -> full Dreyer test pass (5,040 windows): "
              f"{t_fwd / 8 * 5040 / 60:.1f} min   "
              f"one fine-tune epoch (12,392): {t_bwd / 8 * 12392 / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1:] or ["cbramod"])
