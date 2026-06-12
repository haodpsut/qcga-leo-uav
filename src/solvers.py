"""Population metaheuristics with a common interface, batched in PyTorch.

solver(dim, eval_fn, n_part, n_iter, lo, hi, seed, device) -> dict with
  best (dim,), best_fit (float), hist_fit [n_iter], hist_feas [n_iter].

eval_fn(X (P,dim)) -> (fit (P,), info) where info["feasible"] is (P,) bool. All solvers
MAXIMIZE fit. Same signature as qpso.qpso so experiments can swap them freely.

Included: Grey Wolf Optimizer (GWO, the incumbent baseline for joint UAV-LEO from
arXiv 2506.12750), standard PSO, Differential Evolution (DE), and a real-coded Genetic
Algorithm (GA) with adaptive mutation. QPSO lives in qpso.py.
"""

import torch


def _rand(P, dim, gen, device):
    return torch.rand(P, dim, generator=gen).to(device)


def _track(info):
    return info["feasible"].float().mean().item()


def gwo(dim, eval_fn, n_part=64, n_iter=120, lo=-1.0, hi=1.0, seed=0, device="cpu"):
    """Grey Wolf Optimizer. Wolves are pulled toward the best three (alpha, beta, delta);
    the exploration coefficient a anneals 2 -> 0. Maximization variant."""
    gen = torch.Generator("cpu").manual_seed(seed)
    X = lo + (hi - lo) * _rand(n_part, dim, gen, device)
    fit, info = eval_fn(X)
    hist_fit, hist_feas = [], []
    for it in range(n_iter):
        a = 2.0 * (1 - it / max(n_iter - 1, 1))
        top = torch.topk(fit, 3).indices
        leaders = X[top]                                   # (3, dim)
        new = torch.zeros_like(X)
        for j in range(3):
            A = a * (2 * _rand(n_part, dim, gen, device) - 1)
            C = 2 * _rand(n_part, dim, gen, device)
            D = (C * leaders[j].unsqueeze(0) - X).abs()
            new = new + (leaders[j].unsqueeze(0) - A * D)
        X = (new / 3.0).clamp(lo, hi)
        fit, info = eval_fn(X)
        hist_fit.append(fit.max().item()); hist_feas.append(_track(info))
    bi = torch.argmax(fit)
    return {"best": X[bi].clone(), "best_fit": fit[bi].item(),
            "hist_fit": hist_fit, "hist_feas": hist_feas}


def pso(dim, eval_fn, n_part=64, n_iter=120, lo=-1.0, hi=1.0, seed=0, device="cpu",
        w=0.7, c1=1.5, c2=1.5):
    gen = torch.Generator("cpu").manual_seed(seed)
    X = lo + (hi - lo) * _rand(n_part, dim, gen, device)
    V = torch.zeros_like(X)
    fit, info = eval_fn(X)
    pbest, pbest_f = X.clone(), fit.clone()
    gi = torch.argmax(pbest_f); gbest, gbest_f = pbest[gi].clone(), pbest_f[gi].clone()
    hist_fit, hist_feas = [], []
    for it in range(n_iter):
        r1 = _rand(n_part, dim, gen, device); r2 = _rand(n_part, dim, gen, device)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest.unsqueeze(0) - X)
        X = (X + V).clamp(lo, hi)
        fit, info = eval_fn(X)
        imp = fit > pbest_f
        pbest = torch.where(imp.unsqueeze(-1), X, pbest)
        pbest_f = torch.where(imp, fit, pbest_f)
        gi = torch.argmax(pbest_f)
        if pbest_f[gi] > gbest_f:
            gbest, gbest_f = pbest[gi].clone(), pbest_f[gi].clone()
        hist_fit.append(gbest_f.item()); hist_feas.append(_track(info))
    return {"best": gbest, "best_fit": gbest_f.item(),
            "hist_fit": hist_fit, "hist_feas": hist_feas}


def de(dim, eval_fn, n_part=64, n_iter=120, lo=-1.0, hi=1.0, seed=0, device="cpu",
       F=0.6, CR=0.9):
    gen = torch.Generator("cpu").manual_seed(seed)
    X = lo + (hi - lo) * _rand(n_part, dim, gen, device)
    fit, info = eval_fn(X)
    hist_fit, hist_feas = [], []
    for it in range(n_iter):
        idx = torch.stack([torch.randperm(n_part, generator=gen) for _ in range(3)])
        a, b, c = X[idx[0]], X[idx[1]], X[idx[2]]
        mutant = (a + F * (b - c)).clamp(lo, hi)
        cross = _rand(n_part, dim, gen, device) < CR
        trial = torch.where(cross, mutant, X)
        tfit, tinfo = eval_fn(trial)
        better = tfit > fit
        X = torch.where(better.unsqueeze(-1), trial, X)
        fit = torch.where(better, tfit, fit)
        info = tinfo
        hist_fit.append(fit.max().item()); hist_feas.append(_track(info))
    bi = torch.argmax(fit)
    return {"best": X[bi].clone(), "best_fit": fit[bi].item(),
            "hist_fit": hist_fit, "hist_feas": hist_feas}


def ga(dim, eval_fn, n_part=64, n_iter=120, lo=-1.0, hi=1.0, seed=0, device="cpu",
       elite=0.1):
    """Real-coded GA: tournament selection, blend crossover, adaptive Gaussian mutation
    (rate decays with iteration). Tuned to be a fair, non-strawman baseline."""
    gen = torch.Generator("cpu").manual_seed(seed)
    X = lo + (hi - lo) * _rand(n_part, dim, gen, device)
    fit, info = eval_fn(X)
    n_elite = max(1, int(elite * n_part))
    hist_fit, hist_feas = [], []
    for it in range(n_iter):
        order = torch.argsort(fit, descending=True)
        X, fit = X[order], fit[order]
        elites = X[:n_elite]
        # tournament selection (size 3)
        t = torch.randint(0, n_part, (n_part, 3), generator=gen)
        winners = t.gather(1, fit[t].argmax(1, keepdim=True)).squeeze(1)
        parents = X[winners]
        # blend crossover
        perm = torch.randperm(n_part, generator=gen)
        p2 = parents[perm]
        alpha = _rand(n_part, dim, gen, device)
        child = alpha * parents + (1 - alpha) * p2
        # adaptive Gaussian mutation
        rate = 0.3 * (1 - it / max(n_iter - 1, 1)) + 0.02
        sigma = 0.1 * (hi - lo) * (1 - 0.5 * it / max(n_iter - 1, 1))
        mask = (_rand(n_part, dim, gen, device) < rate).to(X.dtype)
        child = (child + mask * sigma * torch.randn(n_part, dim, generator=gen).to(device)).clamp(lo, hi)
        child[:n_elite] = elites                              # elitism
        X = child
        fit, info = eval_fn(X)
        hist_fit.append(fit.max().item()); hist_feas.append(_track(info))
    bi = torch.argmax(fit)
    return {"best": X[bi].clone(), "best_fit": fit[bi].item(),
            "hist_fit": hist_fit, "hist_feas": hist_feas}
