"""Smoke test: QI-CGA vs QI-Euclidean on a small LEO-UAV-ground scenario.

Goal of this first run is NOT publishable numbers, only to prove the pipeline end to
end and to show the expected qualitative signal: the CGA (motor) parameterization
keeps the swarm kinematically feasible by construction, while the Euclidean
parameterization must learn its way out of speed violations.

Run: python experiments/smoke_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cga  # noqa: E402
import decode  # noqa: E402
from qpso import qpso  # noqa: E402
from scenario import Scenario  # noqa: E402


def run(seed=0, device="cpu"):
    scn = Scenario(seed=seed)

    def make_eval(decode_fn):
        def eval_fn(X):
            traj = decode_fn(X, scn)
            return decode.fitness(traj, scn)
        return eval_fn

    dim_cga = scn.M * scn.N * 6
    dim_euc = scn.M * scn.N * 3

    res_cga = qpso(dim_cga, make_eval(decode.cga_decode),
                   lo=-150.0, hi=150.0, seed=seed, device=device)
    res_euc = qpso(dim_euc, make_eval(decode.euclid_decode),
                   lo=-2000.0, hi=2000.0, seed=seed, device=device)
    return scn, res_cga, res_euc


def main():
    torch.set_default_dtype(torch.float64)
    seeds = [0, 1, 2]
    cga_fit, euc_fit, cga_feas, euc_feas = [], [], [], []
    for s in seeds:
        scn, rc, re = run(seed=s)
        cga_fit.append(rc["best_fit"])
        euc_fit.append(re["best_fit"])
        cga_feas.append(rc["hist_feas"][-1])
        euc_feas.append(re["hist_feas"][-1])

    def ms(v):
        t = torch.tensor(v)
        return f"{t.mean():.3f} +/- {t.std():.3f}"

    print("=" * 60)
    print(f"scenario: K={scn.K} users, M={scn.M} UAVs, N={scn.N} slots, "
          f"{len(seeds)} seeds")
    print("-" * 60)
    print(f"best fitness   CGA : {ms(cga_fit)}")
    print(f"best fitness   EUC : {ms(euc_fit)}")
    print(f"final swarm feasibility rate  CGA : {ms(cga_feas)}")
    print(f"final swarm feasibility rate  EUC : {ms(euc_feas)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
