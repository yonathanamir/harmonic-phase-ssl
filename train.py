"""Train and test one embedding on one array.

    python train.py --embedding even-w --filters 2 --data data/linear --out runs/linear/seed0/even-w_D2

Recipe of the report: Adam, constant learning rate 1e-3, batch 8192, cross-entropy, 150 epochs,
samples permuted every epoch, best-validation checkpoint. The run directory gets:

  history.json   one row per epoch          status.json   progress, for the dashboard
  last.pt        resume checkpoint          best.pt       best-validation weights
  results.json   test scores per SNR; written last, so it marks a complete run

A stopped run continues from its last epoch when the same command is given again. On the
linear array, theta and -theta give the same data, so its scores use mirror-folded labels.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from embeddings import EMBEDDINGS
from model import SSLNet
from simulate import SNR_NAMES

N_CLASSES = 72


# ---------------------------------------------------------------- files
def save_json(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    temporary.replace(path)


def save_torch(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


# ---------------------------------------------------------------- data
def to_classes(degrees: torch.Tensor) -> torch.Tensor:
    """1-degree azimuths -> nearest of the 72 5-degree classes (half up, wrap at 360)."""
    return torch.floor(degrees.double() / 5 + 0.5).long() % N_CLASSES


def load_split(paths, with_mag: bool, limit: int = 0):
    """(phase, [mag,] class) tensors from one or more .npz files, in the given order."""
    paths = [paths] if isinstance(paths, Path) else list(paths)
    counts = []
    for p in paths:
        with np.load(p) as z:
            counts.append(len(z["label"]))
    n = min(limit, sum(counts)) if limit else sum(counts)
    keys = ["phase", "mag"] if with_mag else ["phase"]
    out = {k: torch.empty((n, 8, 256), dtype=torch.float32) for k in keys}
    out["label"] = torch.empty(n, dtype=torch.int64)
    filled = 0
    for p, c in zip(paths, counts):
        take = min(c, n - filled)
        if take <= 0:
            break
        with np.load(p) as z:
            for k in keys:
                out[k][filled:filled + take] = torch.from_numpy(z[k][:take])
            out["label"][filled:filled + take] = torch.from_numpy(z["label"][:take].astype(np.int64))
        filled += take
    return (*(out[k] for k in keys), to_classes(out["label"]))


class Batches:
    """Minibatches of in-memory tensors; a dedicated generator gives the epoch permutations."""

    def __init__(self, tensors, batch_size, shuffle=False, seed=0):
        self.tensors, self.batch_size, self.shuffle = tensors, batch_size, shuffle
        self.n = len(tensors[0])
        self.gen = torch.Generator().manual_seed(seed)

    def __iter__(self):
        idx = torch.randperm(self.n, generator=self.gen) if self.shuffle else torch.arange(self.n)
        for lo in range(0, self.n, self.batch_size):
            j = idx[lo:lo + self.batch_size]
            yield tuple(t[j] for t in self.tensors)


# ---------------------------------------------------------------- scores
def fold(y):
    """Mirror-folded class: theta and -theta share one label on the linear array."""
    return torch.minimum(y, (-y) % N_CLASSES)


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    loss, preds, labels = 0.0, [], []
    for *xs, y in batches:
        logits = model(*[x.to(device) for x in xs])
        loss += float(nn.functional.cross_entropy(logits, y.to(device), reduction="sum"))
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
    p, y = torch.cat(preds), torch.cat(labels)
    return p, y, loss / len(y)


def scores(pred, truth, linear, mask=None):
    """Exact accuracy and accuracy within +-5, 10, 25 degrees (folded on the linear array)."""
    if mask is not None:
        pred, truth = pred[mask], truth[mask]
    if not len(truth):
        return {"samples": 0}
    if linear:
        pred, truth = fold(pred), fold(truth)
    dist = (pred - truth).abs()
    dist = torch.minimum(dist, N_CLASSES - dist) * 5
    return {"samples": len(truth), "accuracy": float((dist == 0).double().mean()),
            **{f"within_{d}deg": float((dist <= d).double().mean()) for d in (5, 10, 25)}}


# ---------------------------------------------------------------- training
def run(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    config = {"embedding": args.embedding, "n_filters": args.filters, "data": str(args.data.resolve()),
              "epochs": args.epochs, "batch": args.batch, "lr": 1e-3, "seed": args.seed,
              "n_train": args.train_limit, "projection_chunk": args.projection_chunk}
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != config:
        raise SystemExit(f"{out} holds a run with a different configuration; use another --out")
    save_json(config, out / "config.json")

    paths = sorted(args.data.glob("train*.npz"))
    if not paths:
        raise SystemExit(f"no train*.npz in {args.data}; run generate_data.py first")
    with np.load(paths[0]) as z:
        meta = json.loads(str(z["meta"]))
    linear = bool(np.allclose(np.asarray(meta["mic_xy"])[:, 1], 0.0))
    model = SSLNet(args.embedding, args.filters, projection_chunk=args.projection_chunk).to(device)
    with_mag = model.needs_mag
    train = load_split(paths, with_mag, args.train_limit)
    val = load_split(args.data / "val.npz", with_mag)
    batches = Batches(train, args.batch, shuffle=True, seed=args.seed)
    val_batches = Batches(val, 1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
    params, n_train = sum(p.numel() for p in model.parameters()), len(train[-1])
    metric = "folded accuracy" if linear else "accuracy"

    history, best_state, best_epoch, best_score, elapsed_before = [], None, 0, -1.0, 0.0
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best_state = saved["history"], saved["best_state"]
        best_epoch, best_score, elapsed_before = saved["best_epoch"], saved["best_score"], saved["elapsed"]
        torch.set_rng_state(saved["rng"].cpu())        # map_location moved the states to the device
        if device == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda_rng"]])
        batches.gen.set_state(saved["batch_rng"].cpu())
        print(f"resuming at epoch {len(history) + 1}", flush=True)
    print(f"{args.embedding} (F = {args.filters}) on {meta['array']}: {params:,} parameters, "
          f"{n_train:,} training samples, device {device}", flush=True)
    start = time.time()

    def status(running, epoch):
        spe = history[-1]["secs"] if history else None
        save_json({"arm": out.name, "epoch": epoch, "epochs": args.epochs, "n_train": n_train,
                   "n_classes": N_CLASSES, "params": params, "running": running,
                   "best_epoch": best_epoch, "best_val_acc": best_score,
                   "elapsed_secs": elapsed_before + time.time() - start, "secs_per_epoch": spe,
                   "eta_secs": (args.epochs - epoch) * spe if spe else None}, out / "status.json")

    for epoch in range(len(history) + 1, args.epochs + 1):
        status(True, epoch - 1)
        model.train()
        t0, loss_sum, hits, count = time.time(), 0.0, 0, 0
        for *xs, y in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(*[x.to(device) for x in xs])
            loss = nn.functional.cross_entropy(logits, y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            hits += int((logits.detach().argmax(1).cpu() == y).sum())
            count += len(y)
        p, y, val_loss = predict(model, val_batches, device)
        val_acc = scores(p, y, linear)["accuracy"]
        history.append({"epoch": epoch, "loss": loss_sum / count, "train_acc": hits / count,
                        "val_loss": val_loss, "val_acc": val_acc, "lr": 1e-3,
                        "secs": time.time() - t0})
        if val_acc > best_score:
            best_score, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_torch({"state_dict": best_state, "config": config, "epoch": epoch}, out / "best.pt")
        save_torch({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "history": history, "best_state": best_state, "best_epoch": best_epoch,
                    "best_score": best_score, "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                    "batch_rng": batches.gen.get_state(),
                    "elapsed": elapsed_before + time.time() - start}, out / "last.pt")
        save_json(history, out / "history.json")
        status(True, epoch)
        print(f"epoch {epoch:3d}  loss {history[-1]['loss']:.4f}  val {metric} {val_acc:.4f}  "
              f"({history[-1]['secs']:.0f}s)", flush=True)

    del batches, train, val_batches, val, optimizer            # free memory before testing
    model.load_state_dict(best_state)
    test, endfire = {}, {}
    for snr in SNR_NAMES:
        path = args.data / f"test_snr_{snr}.npz"
        tensors = load_split(path, with_mag)
        p, y, _ = predict(model, Batches(tensors, 1024), device)
        test[snr] = scores(p, y, linear)
        if linear:      # within 20 degrees of the array axis
            with np.load(path) as z:
                axis = torch.from_numpy(((z["theta_deg"] + 90) % 180 - 90).astype(np.float32)).abs()
            endfire[snr] = scores(p, y, True, axis <= 20)
        print(f"test {snr:>4}: {metric} {test[snr]['accuracy']:.4f}, "
              f"within 5 deg {test[snr]['within_5deg']:.4f}", flush=True)
    status(False, args.epochs)
    save_json(config | {"array": meta["array"], "params": params, "best_epoch": best_epoch,
                        "best_val_acc": best_score, "metric": metric, "test": test,
                        "endfire": endfire or None,
                        "elapsed_secs": elapsed_before + time.time() - start}, out / "results.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embedding", choices=EMBEDDINGS, required=True)
    ap.add_argument("--filters", type=int, default=2,
                    help="kernels per phase value, F = D (kernel embeddings only; default 2)")
    ap.add_argument("--data", type=Path, required=True, help="dataset directory of one array")
    ap.add_argument("--out", type=Path, required=True, help="run directory")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--train-limit", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--projection-chunk", type=int, default=512)
    ap.add_argument("--device", choices=("cpu", "cuda"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
