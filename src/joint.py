"""Shared decode + fitness for the joint LEO-UAV problem.

Every solver optimizes the SAME continuous vector of dimension M*N*(2 + Ln):
  - first M*N*2 entries : horizontal position increments (bounded -> kinematically valid)
  - next  M*N*Ln entries: per-slot LEO association logits; association = argmax over the
    VISIBLE satellites (invisible ones masked to -inf), which makes the visibility
    constraint satisfied by construction.

This isolates the optimizer: any performance gap is the search strategy, not the
encoding.
"""

import torch

DIM_POS = 2  # horizontal increment channels (altitude fixed in v1)


def dim_of(scn):
    return scn.M * scn.N * (DIM_POS + scn.Ln)


def decode(X, scn):
    """X (B, dim) -> (traj (B,M,N,3), assoc (B,M,N) long)."""
    B, dev, dt = X.shape[0], X.device, X.dtype
    M, N, Ln = scn.M, scn.N, scn.Ln
    npos = M * N * DIM_POS
    pos_inc = X[:, :npos].view(B, M, N, DIM_POS)
    logits = X[:, npos:].view(B, M, N, Ln)

    # ---- trajectory from bounded horizontal increments ----
    start = scn.start.to(dev, dt)
    traj = torch.zeros(B, M, N, 3, device=dev, dtype=dt)
    traj[:, :, 0, :] = start.view(1, M, 3)
    for n in range(1, N):
        h = pos_inc[:, :, n, :]
        hn = torch.linalg.norm(h, dim=-1, keepdim=True).clamp_min(1e-9)
        h = h * (scn.v_max * torch.tanh(hn / scn.v_max) / hn)
        nxt = traj[:, :, n - 1, :2] + h
        z = torch.full((B, M, 1), scn.alt, device=dev, dtype=dt)
        traj[:, :, n, :] = torch.cat([nxt, z], dim=-1)

    # ---- association = visibility-masked argmax ----
    vis = scn.vis.to(dev, dt).t().view(1, 1, N, Ln)          # (1,1,N,Ln)
    masked = logits.masked_fill(vis.expand(B, M, N, Ln) < 0.5, float("-inf"))
    # if a slot has no visible sat (shouldn't happen), fall back to raw argmax
    no_vis = (vis.sum(-1, keepdim=True) < 0.5)
    masked = torch.where(no_vis.expand_as(masked), logits, masked)
    assoc = masked.argmax(dim=-1)                            # (B,M,N)
    return traj, assoc


def fitness(X, scn, feas_penalty=40.0):
    traj, assoc = decode(X, scn)
    out = scn.evaluate(traj, assoc)
    # soft penalties mirror the hard feasibility flags (vis handled in decode).
    # Penalties are normalized to O(1) per fully-violated point so feas_penalty is the
    # effective per-point cost and dominates any throughput gained inside the no-fly zone.
    steps = traj[:, :, 1:, :] - traj[:, :, :-1, :]
    path = torch.linalg.norm(steps[..., :2], dim=-1)
    speed_v = (path - scn.v_max).clamp_min(0.0).sum(dim=(-1, -2))
    # Penalize a margin-inflated no-fly radius so zero-penalty solutions clear the true
    # boundary with slack (avoids the soft-penalty trap of sitting marginally inside).
    rm = scn.nofly_r.to(X.device, X.dtype) * 1.1
    d2c = ((traj - scn.nofly_c.to(X.device, X.dtype).view(1, 1, 1, 3)) ** 2).sum(-1)
    nofly_v = ((rm ** 2 - d2c).clamp_min(0.0) / rm ** 2).sum(dim=(-1, -2))
    fit = out["objective"] - feas_penalty * (speed_v / scn.v_max + nofly_v)
    return fit, out
