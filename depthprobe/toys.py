"""Analytic stacks with known answers, run through the same metric functions as real models.

Before trusting the E6 metrics on GPT-2 they must tell these apart:
  contraction       h <- (1 - a) h: survival is exactly (1 - a)^(k - l)
  rotation          h <- Q h: nothing is removed, no selectivity
  norm_pump         h <- h + c w: the norm grows, raw survival stays 1 (pure dilution)
  linear_diffusion  Euler steps of a variance-exploding sampler with the exact posterior mean of
                    low-rank Gaussian data: off-manifold noise shrinks by exactly sigma_L / sigma_b
  mixture_diffusion the same with a Gaussian-mixture prior (nonlinear posterior mean)
  manifold_attractor h <- h - a P_off h: denoises (selectivity) but has no noise schedule
  point_attractor   h <- h + a (nearest centre - h): contracts every direction alike
A manifold attractor passes the v1 selectivity test, which is why v1 tests denoising, not diffusion.
"""

import math

import torch

from depthprobe.perturb import propagation_metrics


class ToyStack:
    def __init__(self, name, steps, signal_basis, data):
        self.name, self.steps, self.signal_basis, self.data = name, steps, signal_basis, data

    def run(self, state, start=0):
        states = [state]
        for step in self.steps[start:]:
            states.append(step(states[-1]))
        return states


def _orthonormal(width, rank, generator):
    q, _ = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
    return q[:, :rank], q[:, rank:]


def make_toys(width=32, rank=4, depth=12, seed=0):
    generator = torch.Generator().manual_seed(seed)
    signal, complement = _orthonormal(width, rank, generator)
    scales = torch.linspace(3.0, 1.0, rank, dtype=torch.float64)
    covariance = signal @ torch.diag(scales ** 2) @ signal.T
    sigmas = torch.logspace(math.log10(5.0), math.log10(0.05), depth + 1, dtype=torch.float64)
    centres = torch.randn(3, rank, generator=generator, dtype=torch.float64) @ signal.T * 3.0
    identity = torch.eye(width, dtype=torch.float64)
    toys = []

    toys.append(ToyStack("contraction", [lambda h: 0.8 * h] * depth, signal, None))
    rotation, _ = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
    toys.append(ToyStack("rotation", [lambda h: h @ rotation.T] * depth, signal, None))
    pump = signal[:, 0] * 2.0
    toys.append(ToyStack("norm_pump", [lambda h: h + pump] * depth, signal, None))

    def linear_step(index):
        sigma, following = sigmas[index], sigmas[index + 1]
        posterior = covariance @ torch.linalg.inv(covariance + sigma ** 2 * identity)
        return lambda h: h + (1 - following / sigma) * (h @ posterior.T - h)

    toys.append(ToyStack("linear_diffusion", [linear_step(i) for i in range(depth)], signal, "gaussian"))

    def mixture_step(index):
        sigma, following = sigmas[index], sigmas[index + 1]
        total = covariance + sigma ** 2 * identity
        precision = torch.linalg.inv(total)
        gain = covariance @ precision

        def step(h):
            offsets = h[:, None, :] - centres[None]  # [n, K, d]
            energy = -0.5 * torch.einsum("nkd,de,nke->nk", offsets, precision, offsets)
            weights = torch.softmax(energy, dim=-1)
            means = centres[None] + offsets @ gain.T  # posterior mean under each component
            denoised = (weights[..., None] * means).sum(1)
            return h + (1 - following / sigma) * (denoised - h)
        return step

    toys.append(ToyStack("mixture_diffusion", [mixture_step(i) for i in range(depth)], signal, "mixture"))
    off_projector = complement @ complement.T
    toys.append(ToyStack("manifold_attractor", [lambda h: h - 0.3 * h @ off_projector] * depth, signal, None))

    def point_step(h):
        nearest = centres[torch.cdist(h, centres).argmin(-1)]
        return h + 0.3 * (nearest - h)

    toys.append(ToyStack("point_attractor", [point_step] * depth, signal, "mixture"))
    return toys, {"signal": signal, "complement": complement, "covariance": covariance, "sigmas": sigmas,
                  "centres": centres}


def _initial_states(toy, info, count, generator):
    width = info["signal"].shape[0]
    noise = torch.randn(count, width, generator=generator, dtype=torch.float64)
    if toy.data == "gaussian":
        clean = torch.randn(count, info["covariance"].shape[0], generator=generator, dtype=torch.float64)
        clean = clean @ torch.linalg.cholesky(info["covariance"] + 1e-9 * torch.eye(width, dtype=torch.float64)).T
        return clean + info["sigmas"][0] * noise
    if toy.data == "mixture":
        choice = torch.randint(0, len(info["centres"]), (count,), generator=generator)
        return info["centres"][choice] + noise @ info["signal"] @ info["signal"].T + info["sigmas"][0] * noise
    return torch.randn(count, width, generator=generator, dtype=torch.float64) + 2.0


def toy_signatures(site=3, delta=0.05, count=512, seed=0, width=32, rank=4, depth=12):
    """Mean metrics at the last state for iso, off- and on-manifold injection at `site`, for every toy."""
    toys, info = make_toys(width, rank, depth, seed)
    on_basis, off_basis = info["signal"].float(), info["complement"].float()
    results = {}
    for toy in toys:
        generator = torch.Generator().manual_seed(seed + 1)
        clean = toy.run(_initial_states(toy, info, count, generator))
        z = torch.randn(count, width, generator=generator, dtype=torch.float64)
        row = {}
        for family, basis in (("iso", None), ("off", info["complement"]), ("on", info["signal"])):
            direction = z if basis is None else z @ basis @ basis.T
            direction = direction / direction.norm(dim=-1, keepdim=True)
            eps = delta * clean[site].norm(dim=-1, keepdim=True) * direction
            noisy = toy.run(clean[site] + eps, start=site)
            metrics = propagation_metrics([c.float() for c in clean[site:]], [x.float() for x in noisy], "rms",
                                          off=[off_basis] * len(noisy), on=[on_basis] * len(noisy))
            row[family] = {key: float(values[-1].mean()) for key, values in metrics.items()}
        # Selectivity (content kept minus noise kept) in raw survival, in the primary statistic log A, and in
        # log Lambda = log A - log A0 (normalized absorption beyond carrying the perturbation unchanged).
        row["selectivity"] = row["on"]["log_survival"] - row["off"]["log_survival"]
        row["selectivity_A"] = row["on"]["log_A"] - row["off"]["log_A"]
        row["selectivity_Lambda"] = (row["on"]["log_A"] - row["on"]["log_A0"]) - (row["off"]["log_A"] - row["off"]["log_A0"])
        row["expected_off_survival"] = float(torch.log(info["sigmas"][depth] / info["sigmas"][site]))
        results[toy.name] = row
    return results
