"""Generate the full datasets of the report, for one or both arrays.

Per array (data/<array>/):
  train.npz, val.npz, test_snr_*.npz   one random stream, seed 20260731
                                       (100k training, 20k validation, 36k test per SNR)
  train_s1.npz ... train_s5.npz        five more training shards of 180k, seeds 20260801..5
                                       (1M training samples in total)

About 22 GB per array. Existing complete files are kept, so an interrupted run can be
started again with the same command. --quick writes a tiny set for a smoke test.

    python generate_data.py --arrays circular linear
"""
from __future__ import annotations

import argparse
from pathlib import Path

from simulate import ARRAYS, SNR_NAMES, generate

BASE_SEED = 20260731
FULL = {"n_train": 100_000, "n_val": 20_000, "n_test": 36_000,
        "shards": {f"train_s{i}": (180_000, 20260800 + i) for i in range(1, 6)}}
QUICK = {"n_train": 2_000, "n_val": 500, "n_test": 500, "shards": {}}


def generate_array(out: Path, array: str, quick: bool):
    sizes = QUICK if quick else FULL
    base = ["train.npz", "val.npz", *(f"test_snr_{s}.npz" for s in SNR_NAMES)]
    if not all((out / name).exists() for name in base):
        # The base splits share one random stream, so they are written together.
        print(f"{array}: base splits -> {out}", flush=True)
        generate(out, array, BASE_SEED, sizes["n_train"], sizes["n_val"], sizes["n_test"])
    for name, (count, seed) in sizes["shards"].items():
        if not (out / f"{name}.npz").exists():
            print(f"{array}: {name} ({count:,} samples, seed {seed})", flush=True)
            generate(out, array, seed, count, 0, 0, splits=("train",), train_name=name)
    print(f"{array}: complete", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arrays", nargs="+", choices=list(ARRAYS), default=list(ARRAYS))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--quick", action="store_true", help="tiny datasets for a smoke test")
    args = ap.parse_args()
    for array in args.arrays:
        generate_array(args.out / (f"{array}-quick" if args.quick else array), array, args.quick)


if __name__ == "__main__":
    main()
