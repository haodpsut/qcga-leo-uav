"""LEO-UAV-ground scenario: geometry, channel proxy, objective and constraints.

This is the smoke-test fidelity model. The channel is a free-space-loss proxy here;
the full version will swap in 3GPP TR 38.811 NTN path loss for the UAV-LEO backhaul
and an air-to-ground model for the UAV-user access link.

All evaluation is batched over a population of candidate solutions so the optimizer
can score the whole swarm in one forward pass (CUDA-friendly).
"""

import copy

import torch

import cga


class Scenario:
    def __init__(self, n_users=10, n_uav=2, n_slots=8, area=2000.0, alt=300.0,
                 cov_radius=600.0, v_max=120.0, seed=0,
                 directional=False, cone_deg=60.0):
        g = torch.Generator().manual_seed(seed)
        self.K = n_users
        self.M = n_uav
        self.N = n_slots
        self.area = area
        self.alt = alt
        self.cov_radius = cov_radius  # ground coverage radius of a UAV cell [m]
        self.v_max = v_max            # max horizontal displacement per slot [m]
        # Directional coverage: a UAV serves a user only if the user lies inside the
        # pointing cone (half-angle cone_deg) as well as within range. This makes the
        # UAV heading a real decision variable, so a global rotation of the scene
        # genuinely matters -- the regime where SE(3)-equivariance can pay off.
        self.directional = directional
        self.cone_cos = torch.cos(torch.deg2rad(torch.tensor(cone_deg)))

        # Ground users on a square; z = 0.
        xy = (torch.rand(n_users, 2, generator=g) - 0.5) * 2 * area
        self.users = torch.cat([xy, torch.zeros(n_users, 1)], dim=-1)  # (K, 3)
        self.user_pts = cga.point(self.users)                          # (K, 5)

        # UAV start poses spread across the area, fixed altitude.
        sx = (torch.rand(n_uav, 2, generator=g) - 0.5) * area
        self.start = torch.cat([sx, torch.full((n_uav, 1), alt)], dim=-1)  # (M, 3)

        # One no-fly sphere near the center (kept light for the smoke test).
        self.nofly_c = torch.tensor([[0.0, 0.0, alt]])
        self.nofly_r = torch.tensor([350.0])
        self.nofly = cga.sphere(self.nofly_c, self.nofly_r)  # (1, 5)

        # A single LEO direction (unit vector) with an elevation mask; backhaul cap.
        self.leo_dir = torch.tensor([0.3, 0.2, 0.93])
        self.leo_dir = self.leo_dir / self.leo_dir.norm()
        self.min_elev_cos = torch.cos(torch.deg2rad(torch.tensor(25.0)))  # 25 deg mask
        self.backhaul_cap = 50.0  # aggregate rate cap per UAV via LEO [bps/Hz units]

    def yaw_rotated(self, angle_rad):
        """Return a copy of this scenario with all geometry rotated by yaw about z.

        This is the global SE(3) action (restricted to ground-plane rotation, the
        physically meaningful symmetry since z is up) used for the equivariance study.
        """
        a = torch.as_tensor(angle_rad, dtype=torch.get_default_dtype())
        c, s = torch.cos(a), torch.sin(a)
        Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        new = copy.deepcopy(self)
        new.users = self.users @ Rz.T
        new.user_pts = cga.point(new.users)
        new.start = self.start @ Rz.T
        new.nofly_c = self.nofly_c @ Rz.T
        new.nofly = cga.sphere(new.nofly_c, self.nofly_r)
        new.leo_dir = self.leo_dir @ Rz.T
        return new

    # ---- coverage sphere of a UAV given its 3D position --------------------
    def _cov_sphere(self, pos):
        """pos (..., 3) -> IPNS coverage sphere (..., 5) of radius cov_radius."""
        r = torch.full(pos.shape[:-1], self.cov_radius, device=pos.device, dtype=pos.dtype)
        return cga.sphere(pos, r)

    def evaluate(self, traj, point_dir=None):
        """Score a batch of trajectories.

        traj: (B, M, N, 3) UAV positions per slot.
        point_dir: (B, M, N, 3) UAV pointing directions, required if directional.
        Returns dict with objective (B,) and feasibility mask (B,).
        """
        B = traj.shape[0]
        dev, dt = traj.device, traj.dtype
        users = self.users.to(dev, dt)            # (K, 3)
        user_pts = self.user_pts.to(dev, dt)      # (K, 5)

        # ---- access rate: each user served by best covering UAV per slot ----
        S = self._cov_sphere(traj)                # (B, M, N, 5) coverage spheres
        # <P_k, S_{m,n}> >= 0 means user k is inside UAV (m,n) cell.
        cov = cga.inner(user_pts.view(1, 1, 1, self.K, 5),
                        S.unsqueeze(-2))           # (B, M, N, K)
        inside = cov >= 0

        # Directional gating: user must also lie inside the UAV pointing cone.
        if self.directional:
            assert point_dir is not None, "directional scenario needs point_dir"
            rel = users.view(1, 1, 1, self.K, 3) - traj.unsqueeze(-2)  # (B,M,N,K,3)
            rel = rel / torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-9)
            pd = point_dir / torch.linalg.norm(point_dir, dim=-1, keepdim=True).clamp_min(1e-9)
            align = (pd.unsqueeze(-2) * rel).sum(-1)          # (B,M,N,K) cos angle
            inside = inside & (align >= self.cone_cos.to(dev, dt))

        # Distance-based SNR proxy: SNR ~ 1 / (dist^2). dist user-UAV.
        d2 = ((traj.unsqueeze(-2) - users.view(1, 1, 1, self.K, 3)) ** 2).sum(-1)  # (B,M,N,K)
        snr = 1e6 / (d2 + 1.0)
        rate = torch.log2(1.0 + snr)              # (B, M, N, K)
        rate = torch.where(inside, rate, torch.zeros_like(rate))

        # Best UAV per (slot, user); then backhaul cap per UAV per slot.
        # Sum the access rate each UAV delivers, cap by backhaul, then take best UAV/user.
        served_by_uav = rate.sum(-1)              # (B, M, N) load each UAV carries
        scale = (self.backhaul_cap / served_by_uav.clamp_min(1e-6)).clamp(max=1.0)
        rate = rate * scale.unsqueeze(-1)         # throttle when over backhaul cap
        best_rate = rate.max(dim=1).values        # (B, N, K) best UAV per user
        access = best_rate.sum(dim=(-1, -2))      # (B,) total delivered rate

        # ---- energy proxy: total horizontal path length -----------------
        steps = traj[:, :, 1:, :] - traj[:, :, :-1, :]       # (B, M, N-1, 3)
        path = torch.linalg.norm(steps[..., :2], dim=-1)     # horizontal only
        energy = path.sum(dim=(-1, -2))                      # (B,)

        objective = access - 0.002 * energy

        # ---- feasibility -------------------------------------------------
        # (1) speed limit: every per-slot horizontal step <= v_max.
        speed_ok = (path <= self.v_max + 1e-3).all(dim=(-1, -2))  # (B,)
        # (2) no-fly: UAV must stay outside the no-fly sphere (<P, N> <= 0).
        Pm = cga.point(traj)                                  # (B, M, N, 5)
        nf = cga.inner(Pm, self.nofly.to(dev, dt).view(1, 1, 1, 5))  # (B, M, N)
        nofly_ok = (nf <= 0).all(dim=(-1, -2))
        feasible = speed_ok & nofly_ok

        return {
            "objective": objective,
            "feasible": feasible,
            "speed_ok": speed_ok,
            "nofly_ok": nofly_ok,
            "access": access,
            "energy": energy,
        }
