"""[SUPERSEDED / kept for the record] B2: warm-start across a known 3D rotation.

FINDING that retired this study: the problem's only true symmetry is YAW (rotation
about z), because the UAV kinematics are anisotropic (separate horizontal-speed and
climb-rate limits), so a pitch rotation is NOT a symmetry and no parameterization can
carry the solution across it exactly. Under yaw, the standard Z-Y-X Euler angle is the
OUTER angle and is itself singularity-free, so there is no rotor-vs-Euler gap to exploit
via warm starts. The genuine, clean parameterization advantage is instead shown in
attitude_param_test.py (steep near-nadir pointing, where the Euler search space hits
gimbal lock). The EUC warm numbers here are also unreliable after the fair Euler angle
rescaling (raw-vs-tanh mismatch); do not cite this file's numbers.

Original description below.

B2: warm-start across a known 3D rotation (LEO passing near zenith).

Scenario: elliptical UAV beam, so full attitude (incl. roll) matters. We solve a
canonical epoch from scratch, then move to a new epoch whose geometry is the canonical
one rotated by a KNOWN 3D rotation g about the world y-axis (a pitch sweep; at 90 deg
the look direction passes through zenith, the physically relevant case for an overhead
LEO). By SE(3)-invariance the optimum of the new epoch is g applied to the canonical
solution, so warm-starting should be cheap.

Both methods warm-start WITHIN THEIR NATIVE attitude scheme:
  CGA   : carries g as an initial rotor R0 = g (the per-slot bivector turns are unchanged).
  Euclid: must fold g into its per-slot Euler angles via R_to_euler(g @ R), which is
          ill-conditioned near pitch = 90 deg (gimbal lock).

We report quality at ZERO refine (pure transform) and after a SHORT refine budget. The
position channel transforms identically for both, so any gap is purely the attitude
parameterization.

Run: python experiments/warmstart_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cga  # noqa: E402
import pose  # noqa: E402
from qpso import qpso  # noqa: E402
from scenario import Scenario  # noqa: E402


def solve_cold(scn, decode_fn, seed, n_iter=180, device="cpu"):
    def eval_fn(X):
        p, R = decode_fn(X, scn)
        q = pose.elliptical_quality(scn, p, R)
        return q, {"feasible": torch.ones_like(q, dtype=torch.bool)}
    dim = scn.M * scn.N * 6
    return qpso(dim, eval_fn, n_iter=n_iter, lo=-150.0, hi=150.0, seed=seed, device=device)


def rotate_increments(params, scn, g, channel):
    """Rotate the world-frame position-increment channel of a param vector by g."""
    p = params.view(scn.M, scn.N, 6).clone()
    v = p[:, :, channel:channel + 3]
    p[:, :, channel:channel + 3] = torch.einsum("ij,mnj->mni", g, v)
    return p.reshape(-1)


def quality_of(scn, decode_fn, params, Rleft=None):
    p, R = decode_fn(params.unsqueeze(0), scn, Rleft=Rleft)
    return pose.elliptical_quality(scn, p, R).item()


def refine(scn, decode_fn, warm, seed, Rleft=None, n_iter=20, device="cpu"):
    def eval_fn(X):
        p, R = decode_fn(X, scn, Rleft=Rleft)
        q = pose.elliptical_quality(scn, p, R)
        return q, {"feasible": torch.ones_like(q, dtype=torch.bool)}
    dim = scn.M * scn.N * 6
    res = qpso(dim, eval_fn, n_iter=n_iter, lo=-150.0, hi=150.0, seed=seed,
               device=device, init_center=warm, init_scale=0.05)
    return res["best_fit"]


def main():
    torch.set_default_dtype(torch.float64)
    base = Scenario(seed=0)
    M = base.M
    seeds = [0, 1, 2]
    angles_deg = [0, 30, 60, 80, 90, 100]

    rows = []
    for a_deg in angles_deg:
        a = torch.deg2rad(torch.tensor(float(a_deg)))
        g = cga.rotation3(torch.tensor([0.0, 1.0, 0.0]) * a)  # rotation about world y

        z_cga, z_euc, r_cga, r_euc = [], [], [], []
        for s in seeds:
            c_cga = solve_cold(base, pose.cga_fullpose_decode, seed=s)["best"]
            c_euc = solve_cold(base, pose.euclid_fullpose_decode, seed=s)["best"]

            rscn = base.yaw_rotated(0.0)            # cheap deep copy
            rscn.users = base.users @ g.T
            rscn.user_pts = cga.point(rscn.users)
            rscn.start = base.start @ g.T

            # ---- CGA warm start: Rleft = g, position increments (channel 3) rotated ----
            w_cga = rotate_increments(c_cga, base, g, channel=3)
            Rl = g.view(1, 1, 3, 3).expand(1, M, 3, 3).contiguous()
            z_cga.append(quality_of(rscn, pose.cga_fullpose_decode, w_cga, Rleft=Rl))
            r_cga.append(refine(rscn, pose.cga_fullpose_decode, w_cga, seed=s, Rleft=Rl))

            # ---- Euclid warm start: fold g into Euler angles, increments (ch 0) rotated -
            pe = c_euc.view(M, base.N, 6).clone()
            Rloc = pose.euler_to_R(pe[:, :, 3:6])                 # (M,N,3,3)
            Rnew = g.view(1, 1, 3, 3) @ Rloc
            pe[:, :, 3:6] = pose.R_to_euler(Rnew)
            w_euc = pe.reshape(-1)
            w_euc = rotate_increments(w_euc, base, g, channel=0)
            z_euc.append(quality_of(rscn, pose.euclid_fullpose_decode, w_euc))
            r_euc.append(refine(rscn, pose.euclid_fullpose_decode, w_euc, seed=s))

        def m(v):
            return torch.tensor(v).mean().item()
        rows.append((a_deg, m(z_cga), m(z_euc), m(r_cga), m(r_euc)))

    print("=" * 74)
    print("B2 warm-start under a known pitch rotation (elliptical beam, full attitude)")
    print(f"{len(seeds)} seeds; quality (higher better)")
    print("-" * 74)
    print(f"{'pitch':>6} | {'CGA zero':>9} {'EUC zero':>9} | {'CGA +20it':>10} {'EUC +20it':>10}")
    print("-" * 74)
    for a, zc, ze, rc, re in rows:
        print(f"{a:>5}d | {zc:>9.3f} {ze:>9.3f} | {rc:>10.3f} {re:>10.3f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
