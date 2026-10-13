"""Print the test results of all complete runs as one table per array and seed.

    python summarize.py [--runs runs]

Columns: exact accuracy (%) at every SNR, its mean, and the accuracy within +-5 degrees at 0 dB.
The linear array uses mirror-folded labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulate import SNR_NAMES


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    args = ap.parse_args()
    for root in sorted(p for p in args.runs.glob("*/seed*") if p.is_dir()):
        rows = []
        for d in sorted(p for p in root.iterdir() if (p / "results.json").exists()):
            r = json.loads((d / "results.json").read_text(encoding="utf-8"))
            acc = [100 * r["test"][s]["accuracy"] for s in SNR_NAMES]
            rows.append((d.name, r["params"], acc, 100 * r["test"]["0dB"]["within_5deg"]))
        if not rows:
            continue
        print(f"\n{root.parent.name} array, {root.name}")
        print(f"{'run':16s}{'params':>11s}" + "".join(f"{s:>7s}" for s in SNR_NAMES)
              + f"{'mean':>7s}{'+-5@0dB':>9s}")
        for name, params, acc, tol in sorted(rows, key=lambda row: sum(row[2])):
            print(f"{name:16s}{params:11,d}" + "".join(f"{a:7.1f}" for a in acc)
                  + f"{sum(acc) / len(acc):7.1f}{tol:9.1f}")


if __name__ == "__main__":
    main()
