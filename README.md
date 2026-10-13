# Harmonic phase embeddings for deep sound source localization

Code for the report *Beyond the First Harmonic: Improving Deep Sound Source Localization Using
Harmonic Phase Embeddings* (Yonathan Amir Wangerski, Tel Aviv University, Deep Learning course,
2026).

The von Mises–Bernoulli network (vM-B) of Nakadai et al. (2020) embeds each phase value as
[cos x, sin x], i.e. only the first harmonic. This repository replaces that embedding with learned
von Mises kernels, which contain all harmonics (even, odd or both), with or without magnitude
weighting. Everything else stays as in the original studies: the same simulated sources and the
same fully connected network.

## Contents

| File | Purpose |
|---|---|
| `simulate.py` | Free-field simulation: circular and linear 8-microphone arrays, 100-sinusoid sources with random phases, white noise at a set SNR, FFT phase and magnitude features |
| `embeddings.py` | All embeddings of the report (table below) |
| `model.py` | The network (embedding → Linear → ReLU → BatchNorm → Linear) with the per-bin common-phase augmentation |
| `train.py` | Trains and tests one embedding on one array; resumes after an interruption |
| `generate_data.py`, `generate_data.ps1` | Generates the full datasets |
| `run_campaign.py`, `run_campaign.ps1` | Trains a set of embeddings, arrays, architectures and seeds |
| `dashboard.py`, `dashboard.ps1` | Live training dashboard in the browser |
| `summarize.py` | Prints the test results as tables |
| `setup_env.ps1`, `requirements.txt` | Python environment |

### Embeddings

With c = cos(x − μ), learned centre μ and learned concentration κ per phase value and kernel:

| Name | Embedding | Formula | Features per value |
|---|---|---|---|
| `raw` | raw phase | x | 1 |
| `vmb` | vM-B (first harmonic) | [cos x, sin x] | 2 |
| `vonmises` | von Mises kernel | e^{κc} / e^{κ}, centred | F |
| `even` | even kernel | cosh(κc) / cosh κ, centred | F |
| `odd` | odd kernel | sinh(κc) / sinh κ | F |
| `dual` | dual kernel | (cosh(κₑc) + sinh(κₒc)) / (cosh κₑ + sinh κₒ), centred | F |
| `reim` | real + imaginary (weighted vM-B) | a·[cos x, sin x] | 2 |
| `vonmises-w`, `even-w`, `odd-w`, `dual-w` | magnitude-weighted kernels | a·g(x) | F |

*a* is the magnitude of the same frequency bin. "Centred" means that the exact mean of the
feature for a uniform phase is subtracted. F is the number of kernels per phase value
(`--filters`, default 2). With F = 2, every kernel network has the size of the vM-B network
(about 1.08 M parameters).

## Requirements

- Windows with PowerShell. The Python code also runs on Linux; use the commands in the scripts.
- Python 3.10–3.12.
- An NVIDIA GPU with 24 GB of memory for the full campaign (we used an RTX 3090). The smoke test
  also runs on a CPU.
- 32 GB of RAM: the 1M training samples stay in memory (8 GB of phases, plus 8 GB of
  magnitudes for the weighted embeddings).
- About 45 GB of free disk space for the datasets of both arrays.

## Setup

From a PowerShell prompt in this folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

This creates the environment `.venv` and installs the CUDA 12.8 build of PyTorch and the
requirements. For a CPU-only environment, add `-Cuda cpu`. All other scripts use `.venv`.

If PowerShell does not allow scripts, run this once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Then you can start the scripts directly,
as in the commands below.

## Smoke test (a few minutes)

```powershell
.\generate_data.ps1 -Quick
.\run_campaign.ps1 -Quick
.\.venv\Scripts\python.exe summarize.py --runs runs-quick
```

This trains all embeddings on both arrays for 2 epochs on tiny datasets. The accuracies are near
chance; the test only shows that everything runs.

## Full campaign

### 1. Generate the datasets

```powershell
.\generate_data.ps1                      # both arrays
.\generate_data.ps1 -Arrays circular     # or one array: circular or linear
```

Per array, this writes to `data\<array>\` a training set of 1M samples (six files), 20k
validation samples and 36k test samples for each SNR (clean, 40, 30, 20, 10 and 0 dB). The
seeds are fixed, so the data are the same on every computer. If the generation stops, give the
same command again: complete files are kept.

### 2. Train

```powershell
.\run_campaign.ps1
```

By default this trains all 11 embeddings on both arrays, with F = 2 and seed 0: 150 epochs, Adam
with a constant learning rate of 1e-3, batches of 8192, and the best-validation checkpoint.
Options:

| Option | Example | Meaning |
|---|---|---|
| `-Arrays` | `-Arrays linear` | arrays: `circular`, `linear` |
| `-Embeddings` | `-Embeddings vmb,even,even-w` | embeddings (names in the table above) |
| `-Filters` | `-Filters 2,8` | architectures: kernels per value, F = D |
| `-Seeds` | `-Seeds 0,1,2` | training seeds |
| `-Epochs` | `-Epochs 150` | epochs per run |

The runs are done one after the other. On an RTX 3090 a run takes 20–60 minutes, so the full
campaign (22 runs) takes about half a day. If the campaign stops, give the same command again:
complete runs are skipped, and a stopped run continues from its last epoch with the same result
as an uninterrupted run.

Each run writes to `runs\<array>\seed<seed>\<run>\` (for example `runs\linear\seed0\even-w_D2`):
`history.json` (one row per epoch), `best.pt` (weights), and `results.json` (test scores, written
last).

### 3. Watch the training

In a second PowerShell window:

```powershell
.\dashboard.ps1
```

This opens `http://127.0.0.1:8770/` in the browser. The page shows all planned runs, their
progress and the learning curves, and updates every 5 seconds.

### 4. Results

```powershell
.\.venv\Scripts\python.exe summarize.py
```

This prints one table per array and seed: the exact accuracy (%) at each SNR, the mean over the
SNRs, and the accuracy within ±5° at 0 dB. On the linear array, θ and −θ give the same data, so
its scores use mirror-folded labels (37 classes).

## Single run

```powershell
.\.venv\Scripts\python.exe train.py --embedding dual-w --filters 2 --data data\linear --out runs\test\dual-w_D2
```

`python train.py --help` lists all options.

## References

- K. Nakadai et al., "Sound source localization based on von-Mises–Bernoulli deep neural
  network," *Proc. IEEE/SICE SII*, 2020.
- K. Itoyama et al., "Assessment of von Mises–Bernoulli deep neural network in sound source
  localization," *Proc. Interspeech*, 2021.
