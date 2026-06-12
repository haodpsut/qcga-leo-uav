"""Quantum-behaved Particle Swarm Optimization (QPSO), batched in PyTorch.

QPSO (Sun et al., 2004) is a velocity-free, quantum-inspired PSO: each particle is
attracted toward a local point p between its own best and the global best, and is
displaced using the mean-best position with a logarithmic quantum kernel. We use it
as the shared optimizer so that the only thing that changes between the CGA run and
the Euclidean run is the decode (the search parameterization), which is the clean
ablation for the CGA contribution.
"""

import torch


def qpso(dim, eval_fn, n_part=64, n_iter=120, lo=-1.0, hi=1.0, seed=0, device="cpu",
         init_center=None, init_scale=0.1):
    """Minimization-free QPSO that MAXIMIZES eval_fn.

    eval_fn(X) -> (fit (P,), info dict) where higher fit is better. info must carry a
    boolean "feasible" (P,) used only for reporting the swarm feasibility rate.

    Returns dict with best params, best fit, and per-iteration history of best fit and
    swarm feasibility rate (for convergence and feasibility-rate plots).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if init_center is None:
        X = lo + (hi - lo) * torch.rand(n_part, dim, generator=g).to(device)
    else:
        # Warm start: cluster the swarm around a provided solution (one particle is
        # exactly the warm point) so the optimizer only has to refine locally.
        c = init_center.to(device).view(1, dim)
        noise = init_scale * (hi - lo) * torch.randn(n_part, dim, generator=g).to(device)
        X = c + noise
        X[0] = c[0]

    fit, info = eval_fn(X)
    pbest = X.clone()
    pbest_fit = fit.clone()
    gidx = torch.argmax(pbest_fit)
    gbest = pbest[gidx].clone()
    gbest_fit = pbest_fit[gidx].clone()

    hist_fit, hist_feas = [], []
    for it in range(n_iter):
        # Contraction-expansion coefficient anneals from 1.0 to 0.5.
        beta = 1.0 - 0.5 * it / max(n_iter - 1, 1)
        mbest = pbest.mean(dim=0, keepdim=True)                     # (1, dim)
        phi = torch.rand(n_part, dim, generator=g).to(device)
        p = phi * pbest + (1 - phi) * gbest.unsqueeze(0)            # local attractor
        u = torch.rand(n_part, dim, generator=g).to(device).clamp_min(1e-12)
        sign = torch.where(torch.rand(n_part, dim, generator=g).to(device) < 0.5,
                           -1.0, 1.0)
        X = p + sign * beta * (mbest - X).abs() * torch.log(1.0 / u)

        fit, info = eval_fn(X)
        improved = fit > pbest_fit
        pbest = torch.where(improved.unsqueeze(-1), X, pbest)
        pbest_fit = torch.where(improved, fit, pbest_fit)
        gidx = torch.argmax(pbest_fit)
        if pbest_fit[gidx] > gbest_fit:
            gbest = pbest[gidx].clone()
            gbest_fit = pbest_fit[gidx].clone()

        hist_fit.append(gbest_fit.item())
        hist_feas.append(info["feasible"].float().mean().item())

    return {
        "best": gbest,
        "best_fit": gbest_fit.item(),
        "hist_fit": hist_fit,
        "hist_feas": hist_feas,
    }
