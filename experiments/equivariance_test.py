"""Equivariance study: does CGA's motor search stay invariant under scene rotation?

Setup uses DIRECTIONAL coverage (a pointing cone) so the UAV heading is a real
decision variable and a global yaw rotation of the scene genuinely changes the
problem. We then ask two honest questions:

  (A) Re-solve invariance. Rotate the whole scene by yaw angle g, re-optimize from
      scratch with the same budget, and measure the best objective. An equivariant
      search should give the SAME quality for every g (low spread across rotations);
      an orientation-sensitive parameterization will vary more.

  (B) Transfer / zero-cost reuse. Solve the canonical scene once, then apply the same
      yaw to the found solution and evaluate on the rotated scene WITHOUT re-optimizing.
      CGA transforms the motors exactly (heading is intrinsic), so quality should be
      retained. The Euclidean baseline carries pointing as a free world-frame vector;
      we rotate it the obvious way and see how much quality survives.

Run: python experiments/equivariance_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import decode  # noqa: E402
from qpso import qpso  # noqa: E402
from scenario import Scenario  # noqa: E402


def solve(scn, decode_pose_fn, seed, n_iter=150, lo=-150.0, hi=150.0, device="cpu"):
    def eval_fn(X):
        traj, pdir = decode_pose_fn(X, scn)
        return decode.fitness(traj, scn, point_dir=pdir)
    dim = scn.M * scn.N * 6
    res = qpso(dim, eval_fn, n_iter=n_iter, lo=lo, hi=hi, seed=seed, device=device)
    return res


def quality(scn, decode_pose_fn, params):
    traj, pdir = decode_pose_fn(params.unsqueeze(0), scn)
    out = scn.evaluate(traj, point_dir=pdir)
    return out["objective"].item()


def main():
    torch.set_default_dtype(torch.float64)
    angles = [0.0, 0.5, 1.0, 1.5708, 2.5, 3.1416]  # yaw rotations [rad]
    seeds = [0, 1, 2]

    methods = {
        "CGA": decode.cga_decode_pose,
        "EUC": decode.euclid_incr_decode_pose,
    }

    # (A) re-solve invariance: quality on each rotated scene, solved from scratch.
    resolve = {m: [] for m in methods}
    # (B) transfer: solve canonical, evaluate the SAME params on the rotated scene.
    transfer = {m: [] for m in methods}

    base = Scenario(directional=True, seed=0)
    for s in seeds:
        canon_sol = {m: solve(base, fn, seed=s)["best"] for m, fn in methods.items()}
        for a in angles:
            rot = base.yaw_rotated(a)
            for m, fn in methods.items():
                resolve[m].append(solve(rot, fn, seed=s)["best_fit"])
                # transfer canonical params unchanged: a truly equivariant search in an
                # intrinsic frame should not even need to transform them.
                transfer[m].append(quality(rot, fn, canon_sol[m]))

    def ms(v):
        t = torch.tensor(v)
        return f"{t.mean():7.3f} +/- {t.std():6.3f}"

    print("=" * 66)
    print("DIRECTIONAL coverage; yaw rotations:", [round(a, 2) for a in angles])
    print(f"{len(seeds)} seeds x {len(angles)} rotations")
    print("-" * 66)
    print("(A) re-solve from scratch on each rotated scene  [quality across rot]")
    for m in methods:
        print(f"    {m}: {ms(resolve[m])}")
    print("(B) transfer canonical solution to rotated scene [quality retained]")
    for m in methods:
        print(f"    {m}: {ms(transfer[m])}")
    print("=" * 66)


if __name__ == "__main__":
    main()
