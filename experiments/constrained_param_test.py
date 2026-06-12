"""Make-or-break test for the reframed paper.

Two questions, answered honestly:

  (a) Does the singularity-free pose win belong to CGA specifically, or to the whole
      geometric-algebra / quaternion family? We compare rotor (CGA), quaternion, and
      Euler attitudes on the steep-pointing constrained problem. Expectation: CGA and
      quaternion tie, both beat Euler. If so, we must NOT claim the win as CGA-specific.

  (b) Does the CGA conformal constraint layer compute anything a standard vector
      implementation does not? We evaluate the no-fly penalty both ways (conformal inner
      product vs plain Euclidean) and report the maximum absolute difference. Expectation:
      ~0, i.e. the CGA constraint layer is a notational unification, not a different or
      better computation. We state this plainly rather than dress it up.

Run: python experiments/constrained_param_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pose  # noqa: E402
from qpso import qpso  # noqa: E402
from scenario import Scenario  # noqa: E402


def cold(scn, decode_fn, atti_dim, seed, n_iter=200):
    def eval_fn(X):
        p, R = decode_fn(X, scn)
        q = pose.constrained_quality(scn, p, R, mode="cga")
        return q, {"feasible": torch.ones_like(q, dtype=torch.bool)}
    dim = scn.M * scn.N * (3 + atti_dim)
    return qpso(dim, eval_fn, n_iter=n_iter, lo=-150.0, hi=150.0, seed=seed)


def main():
    torch.set_default_dtype(torch.float64)
    scn = Scenario(n_users=12, n_uav=2, n_slots=8, area=500.0, alt=300.0,
                   cov_radius=700.0, seed=0)
    seeds = [0, 1, 2, 3, 4]

    methods = [
        ("CGA-rotor", pose.cga_fullpose_decode, 3),
        ("Quaternion", pose.quat_fullpose_decode, 4),
        ("Euler", pose.euclid_fullpose_decode, 3),
    ]
    res = {name: [] for name, _, _ in methods}
    for s in seeds:
        for name, fn, ad in methods:
            res[name].append(cold(scn, fn, ad, seed=s)["best_fit"])

    def ms(v):
        t = torch.tensor(v); return f"{t.mean():7.3f} +/- {t.std():6.3f}"

    # (b) constraint-layer equivalence: CGA inner product vs plain vector algebra.
    g = torch.Generator().manual_seed(7)
    rnd = (torch.rand(50, scn.M, scn.N, 3, generator=g) - 0.5) * 1500.0
    rnd[..., 2] = 300.0
    v_cga = pose.nofly_penalty(scn, rnd, mode="cga")
    v_vec = pose.nofly_penalty(scn, rnd, mode="vec")
    max_diff = (v_cga - v_vec).abs().max().item()

    print("=" * 62)
    print("(a) steep-pointing constrained coverage; cold solve; 5 seeds")
    print("-" * 62)
    for name, _, _ in methods:
        print(f"    {name:12s}: {ms(res[name])}")
    print("-" * 62)
    print("(b) no-fly penalty: CGA inner product vs plain vector algebra")
    print(f"    max |CGA - vector| over 50 random configs = {max_diff:.3e}")
    print("=" * 62)


if __name__ == "__main__":
    main()
