"""Phase embeddings g(x), applied to each of the J = 8 x 256 phase values of a sample.

  name         features per value (D)   formula, with c_f = cos(x - mu_f)
  raw          1                        x
  vmb          2                        [cos x, sin x]                          (vM-B, first harmonic)
  vonmises     F                        e^{k c} / e^{k}                          - mean
  even         F                        cosh(k c) / cosh k                       - mean
  odd          F                        sinh(k c) / sinh k                       (zero mean)
  dual         F                        (cosh(ke c) + sinh(ko c)) / (cosh ke + sinh ko) - mean
  reim         2                        a [cos x, sin x]                         (weighted vM-B = real+imaginary)
  <kernel>-w   F                        a g(x)                                   (magnitude-weighted kernel)

a is the magnitude of the same frequency bin. Every kernel has a learned centre mu and a
learned concentration k = softplus(rho) > 0 per value and filter (the dual kernel has two),
and is normalized to a peak of 1. "mean" is the exact mean of the feature for a uniform phase,
I0(k) e^{-k}, I0(k) / cosh k and I0(ke) / (cosh ke + sinh ko); subtracting it keeps the
ReLU-before-BatchNorm network from making hidden units dead or always on. All forms keep the
exponents <= 0, so no kappa can overflow.
"""
from __future__ import annotations

import torch
from torch import nn

KERNELS = ("vonmises", "even", "odd", "dual")
EMBEDDINGS = ("raw", "vmb", *KERNELS, "reim", *(f"{k}-w" for k in KERNELS))
KAPPA_INIT = (0.5, 3.0)


def inverse_softplus(y):
    return y + torch.log(-torch.expm1(-y))


class RawPhase(nn.Module):
    needs_mag, d = False, 1

    def forward(self, x):
        return x


class FirstHarmonic(nn.Module):
    """vM-B: [cos x, sin x]; the following linear layer supplies A_i and B_i."""
    needs_mag, d = False, 2

    def forward(self, x):
        return torch.cat([torch.cos(x), torch.sin(x)], dim=1)


class RealImaginary(nn.Module):
    """[a cos x, a sin x]: the magnitude-weighted vM-B."""
    needs_mag, d = True, 2

    def forward(self, x, mag):
        return torch.cat([mag * torch.cos(x), mag * torch.sin(x)], dim=1)


class KernelBank(nn.Module):
    """F learned kernels of one type per phase value; output (B, J * F), index j * F + f."""
    needs_mag = False

    def __init__(self, kind: str, n_inputs: int, n_filters: int):
        super().__init__()
        if kind not in KERNELS:
            raise ValueError(kind)
        self.kind, self.d = kind, n_filters
        shape = (n_inputs, n_filters)
        self.mu = nn.Parameter(2 * torch.pi * torch.rand(shape))
        first = "_ko" if kind == "odd" else "_ke"
        setattr(self, first, nn.Parameter(inverse_softplus(torch.empty(shape).uniform_(*KAPPA_INIT))))
        if kind == "dual":
            self._ko = nn.Parameter(inverse_softplus(torch.empty(shape).uniform_(*KAPPA_INIT)))

    @property
    def kappa_e(self):
        """Concentration of the even part (von Mises: the single kappa; odd: zero)."""
        if self.kind == "odd":
            return torch.zeros_like(self._ko)
        return nn.functional.softplus(self._ke)

    @property
    def kappa_o(self):
        """Concentration of the odd part (von Mises: the single kappa; even: zero)."""
        if self.kind == "vonmises":
            return self.kappa_e
        if self.kind == "even":
            return torch.zeros_like(self._ke)
        return nn.functional.softplus(self._ko)

    def kernel(self, x):
        c = torch.cos(x[:, :, None] - self.mu)                        # (B, J, F)
        ke, ko = self.kappa_e, self.kappa_o
        if self.kind == "vonmises":
            return torch.exp(ke * (c - 1.0))
        if self.kind == "odd":
            num = torch.expm1(ko * (c - 1.0)) - torch.expm1(-ko * (c + 1.0))
            return num / -torch.expm1(-2.0 * ko)
        m = torch.maximum(ke, ko)                                     # even and dual
        num = (torch.exp(ke * c - m) + torch.exp(-ke * c - m)
               + torch.exp(ko * c - m) - torch.exp(-ko * c - m))
        den = (torch.exp(ke - m) + torch.exp(-ke - m)
               + torch.exp(ko - m) - torch.exp(-ko - m))
        return num / den

    def uniform_mean(self):
        """Exact mean of each feature for a uniform phase (the zeroth harmonic), (J, F)."""
        ke, ko = self.kappa_e, self.kappa_o
        if self.kind == "odd":
            return torch.zeros_like(ko)
        if self.kind == "vonmises":
            return torch.special.i0e(ke)                              # I0(k) e^{-k}
        m = torch.maximum(ke, ko)
        num = torch.special.i0e(ke) * torch.exp(ke - m)
        den = 0.5 * (torch.exp(ke - m) + torch.exp(-ke - m)
                     + torch.exp(ko - m) - torch.exp(-ko - m))
        return num / den

    def forward(self, x):
        features = self.kernel(x).flatten(1)
        if self.kind == "odd":
            return features
        return features - self.uniform_mean().flatten()


class MagnitudeWeighted(nn.Module):
    """a g(x): every feature of a value multiplied by the magnitude of that value."""
    needs_mag = True

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base, self.d = base, base.d

    def forward(self, x, mag):
        return self.base(x) * mag.repeat_interleave(self.d, dim=1)


def make_embedding(name: str, n_inputs: int = 2048, n_filters: int = 2) -> nn.Module:
    """n_filters = F, the number of kernels per value (D = F features); 2 matches vM-B's size."""
    if name == "raw":
        return RawPhase()
    if name == "vmb":
        return FirstHarmonic()
    if name == "reim":
        return RealImaginary()
    if name in KERNELS:
        return KernelBank(name, n_inputs, n_filters)
    if name.endswith("-w") and name[:-2] in KERNELS:
        return MagnitudeWeighted(KernelBank(name[:-2], n_inputs, n_filters))
    raise ValueError(f"unknown embedding {name!r}; choose from {', '.join(EMBEDDINGS)}")
