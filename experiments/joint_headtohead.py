"""Make-or-break head-to-head on the joint LEO-UAV trajectory + association problem.

QPSO (the proposed quantum-inspired solver) vs the incumbent Grey Wolf Optimizer
(arXiv 2506.12750) and the classical metaheuristic suite (PSO, DE, GA), all optimizing
the SAME decision vector with the SAME budget. Reports final objective, throughput,
feasibility, and convergence at a mid-iteration checkpoint, mean +/- std over seeds.

Run: python experiments/joint_headtohead.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joint  # noqa: E402
import solvers  # noqa: E402
from qpso import qpso  # noqa: E402
from scenario_joint import JointScenario  # noqa: E402


def main():
    torch.set_default_dtype(torch.float64)
    seeds = [0, 1, 2, 3, 4]
    n_part, n_iter = 60, 150

    methods = {
        "QPSO": qpso,
        "GWO": solvers.gwo,
        "PSO": solvers.pso,
        "DE": solvers.de,
        "GA": solvers.ga,
    }
    obj = {m: [] for m in methods}
    thr = {m: [] for m in methods}
    feas = {m: [] for m in methods}
    mid = {m: [] for m in methods}

    for s in seeds:
        scn = JointScenario(seed=s)
        dim = joint.dim_of(scn)

        def eval_fn(X):
            return joint.fitness(X, scn)

        for name, fn in methods.items():
            r = fn(dim, eval_fn, n_part=n_part, n_iter=n_iter,
                   lo=-200.0, hi=200.0, seed=s)
            # re-evaluate best to recover throughput + feasibility metrics
            _, info = joint.fitness(r["best"].unsqueeze(0), scn)
            obj[name].append(r["best_fit"])
            thr[name].append(info["throughput"].item())
            feas[name].append(float(info["feasible"].item()))
            mid[name].append(r["hist_fit"][n_iter // 3])

    def ms(v):
        t = torch.tensor(v); return f"{t.mean():8.2f} +/- {t.std():6.2f}"

    print("=" * 70)
    print(f"Joint LEO-UAV: K={scn.K} M={scn.M} N={scn.N} Ln={scn.Ln}; "
          f"{len(seeds)} seeds, {n_part}x{n_iter}")
    print("-" * 70)
    print(f"{'solver':6} | {'objective':>20} | {'throughput[Mbps]':>20} | {'feas':>5}")
    print("-" * 70)
    for m in methods:
        print(f"{m:6} | {ms(obj[m]):>20} | {ms(thr[m]):>20} | "
              f"{torch.tensor(feas[m]).mean():.2f}")
    print("-" * 70)
    print("convergence (objective at iter n/3):")
    for m in methods:
        print(f"  {m:6}: {ms(mid[m])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
