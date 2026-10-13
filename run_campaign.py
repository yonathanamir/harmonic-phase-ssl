"""Train every requested embedding on every requested array, one run after the other.

    python run_campaign.py                                     # the full campaign of the report
    python run_campaign.py --arrays linear --embeddings vmb even dual-w --filters 2 8

Runs go to runs/<array>/seed<seed>/<run>, where <run> is the embedding name, plus _D<F> for
the kernel embeddings (raw, vmb and reim have a fixed D and do not depend on --filters). A
complete run (results.json present) is skipped and a stopped run resumes, so the same command
continues an interrupted campaign. Each run is a separate process, which frees its memory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from embeddings import EMBEDDINGS, KERNELS

FIXED_D = ("raw", "vmb", "reim")


def run_name(embedding: str, filters: int) -> str:
    return embedding if embedding in FIXED_D else f"{embedding}_D{filters}"


def plan(embeddings, filters):
    """(run name, embedding, F) in order, each run once."""
    runs = {}
    for f in filters:
        for e in embeddings:
            runs.setdefault(run_name(e, f), (e, f))
    return [(name, e, f) for name, (e, f) in runs.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arrays", nargs="+", choices=("circular", "linear"),
                    default=["circular", "linear"])
    ap.add_argument("--embeddings", nargs="+", choices=EMBEDDINGS, default=list(EMBEDDINGS))
    ap.add_argument("--filters", nargs="+", type=int, default=[2],
                    help="architectures to train: kernels per value, F = D (default 2)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--quick", action="store_true",
                    help="smoke test on the --quick datasets: 2 epochs, small batches")
    args = ap.parse_args()

    extra = []
    if args.quick:
        args.epochs, args.runs = 2, Path(str(args.runs) + "-quick")
        extra = ["--batch", "512", "--train-limit", "2000"]
    runs = plan(args.embeddings, args.filters)
    failed = []
    for seed in args.seeds:
        for array in args.arrays:
            data = args.data / (f"{array}-quick" if args.quick else array)
            root = args.runs / array / f"seed{seed}"
            root.mkdir(parents=True, exist_ok=True)
            # The dashboard reads the plan from campaign.json, so queued runs are visible too.
            (root / "campaign.json").write_text(json.dumps({
                "label": f"{array} array, seed {seed}", "arms": [name for name, _, _ in runs],
                "epochs": args.epochs}, indent=1), encoding="utf-8")
            for name, embedding, f in runs:
                out = root / name
                if (out / "results.json").exists():
                    print(f"complete: {out}", flush=True)
                    continue
                print(f"\n=== {array}, seed {seed}: {name}", flush=True)
                code = subprocess.call([sys.executable, "-u", "train.py", "--embedding", embedding,
                                        "--filters", str(f), "--data", str(data), "--out", str(out),
                                        "--epochs", str(args.epochs), "--seed", str(seed), *extra])
                if code != 0:
                    failed.append(str(out))
                    print(f"FAILED ({code}): {out}; the campaign continues", flush=True)
    if failed:
        print("\nFailed runs (run the same command again to retry):\n  " + "\n  ".join(failed))
        return 1
    print("\nCampaign complete. Summary: python summarize.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
