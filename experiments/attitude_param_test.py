"""Clean decision test: is the rotor attitude parameterization a better SEARCH SPACE
than minimal Euler angles for an elliptical-beam coverage problem, when good solutions
require steep (near-nadir) pointing that sits near the Euler gimbal-lock singularity?

No warm start, no scene rotation. Same QPSO, same budget, same fair angle scaling
(Euler mapped to ~(-pi, pi)). The ONLY difference is rotor turns vs Euler angles for
attitude. This isolates the parameterization quality question that the warm-start study
could not answer cleanly (the problem's only symmetry is yaw, where Euler is fine too).

Run: python experiments/attitude_param_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pose  # noqa: E402
from qpso import qpso  # noqa: E402
from scenario import Scenario  # noqa: E402


def cold(scn, decode_fn, seed, n_iter=200):
    def eval_fn(X):
        p, R = decode_fn(X, scn)
        q = pose.elliptical_quality(scn, p, R)
        return q, {"feasible": torch.ones_like(q, dtype=torch.bool)}
    dim = scn.M * scn.N * 6
    return qpso(dim, eval_fn, n_iter=n_iter, lo=-150.0, hi=150.0, seed=seed)


def main():
    torch.set_default_dtype(torch.float64)
    # Tight users near the UAVs at altitude => boresight must point steeply down,
    # i.e. near the Euler pitch = -90 deg singularity.
    scn = Scenario(n_users=12, n_uav=2, n_slots=8, area=500.0, alt=300.0,
                   cov_radius=700.0, seed=0)
    seeds = [0, 1, 2, 3, 4]
    cga_q, euc_q, cga_60, euc_60 = [], [], [], []
    for s in seeds:
        rc = cold(scn, pose.cga_fullpose_decode, seed=s)
        re = cold(scn, pose.euclid_fullpose_decode, seed=s)
        cga_q.append(rc["best_fit"]); euc_q.append(re["best_fit"])
        cga_60.append(rc["hist_fit"][60]); euc_60.append(re["hist_fit"][60])

    def ms(v):
        t = torch.tensor(v); return f"{t.mean():7.3f} +/- {t.std():6.3f}"
    print("=" * 60)
    print("Steep-pointing elliptical coverage; cold solve; 5 seeds")
    print("-" * 60)
    print(f"final quality   CGA(rotor) : {ms(cga_q)}")
    print(f"final quality   EUC(euler) : {ms(euc_q)}")
    print(f"quality @ iter60 CGA       : {ms(cga_60)}")
    print(f"quality @ iter60 EUC       : {ms(euc_60)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
