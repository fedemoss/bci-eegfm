"""Join the subject-grouped validation scores (from the run logs) with the
test balanced accuracy (from benchopt's parquet) into one table.

The objective only records the test metric, and the val score is what model
selection is allowed to look at — so the two have to be stitched together
here rather than read from a single file.
"""

import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

HOME = Path(os.environ.get("EEGFM_HOME", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("WORK", HOME / "work"))
OUTPUTS = Path(os.environ.get("BENCH", WORK / "input/2026-competition/tracks/bci_decoding")) / "outputs"
LOGS = Path(os.environ.get("LOGS", WORK / "output/logs"))

_PARAM = re.compile(r"(\w+)=([^,\]]+)")
_ARM = re.compile(r"^\[CBraMod\] arm=(\S+) init=(\S+) head=(\S+) gain=(\S+)")
_BEST = re.compile(r"^\[CBraMod\] best val_bal_acc=([\d.]+)")


def parse_logs():
    """(arm, init, head, gain) -> best grouped-val balanced accuracy."""
    out = {}
    for path in sorted(LOGS.glob("*.log")):
        key = None
        for line in path.read_text(errors="ignore").splitlines():
            m = _ARM.match(line)
            if m:
                key = m.groups()
            m = _BEST.match(line)
            if m and key:
                out[key] = float(m.group(1))
    return out


def parse_parquet():
    rows = []
    for f in sorted(glob.glob(str(OUTPUTS / "*.parquet")), key=os.path.getmtime):
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            name = str(r["solver_name"])
            if not name.startswith("EEGFM"):
                continue
            p = dict(_PARAM.findall(name))
            rows.append({
                "arm": p.get("arm"), "init": p.get("init"),
                "head": p.get("head"), "gain": p.get("gain"),
                "adapt": p.get("adapt"), "tent_div": p.get("tent_diversity"),
                "ssl": p.get("ssl_tasks"),
                "test_bal_acc": r["objective_balanced_accuracy"],
                "time_s": round(r["time"]),
                "run": os.path.basename(f),
            })
    return pd.DataFrame(rows)


def main():
    df = parse_parquet()
    if df.empty:
        print("no CBraMod rows in", OUTPUTS)
        return 1
    val = parse_logs()
    df["val_bal_acc"] = [
        val.get((r.arm, r.init, r.head, r.gain)) for r in df.itertuples()
    ]
    df = df.drop_duplicates(
        subset=["arm", "init", "head", "gain", "adapt", "tent_div", "ssl"],
        keep="last"
    ).sort_values("test_bal_acc", ascending=False)
    cols = ["arm", "init", "head", "gain", "ssl", "adapt", "tent_div",
            "val_bal_acc", "test_bal_acc", "time_s"]
    print(df[cols].to_string(index=False, na_rep="-"))
    print("\nReference on the same split: EEGNet 0.7651, MyModel 0.7794, chance 0.50")
    print("Select on val_bal_acc (subject-grouped). test_bal_acc is the "
          "warm-up leaderboard metric — read it, do not tune on it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
