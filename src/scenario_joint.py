"""Joint LEO-UAV trajectory + association scenario (Transactions-grade core).

Integrated SAGIN setting: ground users served by UAV aerial base stations, each UAV
backhauled to one LEO satellite per slot (association + handover). Decision = UAV
trajectory (bounded horizontal increments, fixed altitude in v1) + per-slot LEO
association. All solvers optimize the SAME continuous decision vector so the comparison
isolates the optimizer.

Channel (documented, TR 38.811-style):
  - UAV-LEO backhaul: free-space path loss at Ka-band + elevation-dependent slant range
    + log-normal shadow fading; visibility gated by a minimum elevation angle.
  - UAV-user access: air-to-ground free-space loss at S-band; coverage by range.
Numbers are representative, not calibrated to a specific deployment; the relative
comparison between solvers is the object of study.

Batched over a population B for swarm evaluation.
"""

import math

import torch

R_E = 6371e3      # Earth radius [m]
H_LEO = 600e3     # LEO altitude [m]
C0 = 299792458.0


def _fspl_db(d_m, f_hz):
    """Free-space path loss [dB] for distance d_m [m] at frequency f_hz [Hz]."""
    return 20.0 * torch.log10(d_m.clamp_min(1.0)) + 20.0 * math.log10(f_hz) \
        + 20.0 * math.log10(4.0 * math.pi / C0)


def _slant_range(elev_rad):
    """Slant range UAV->LEO [m] from elevation angle, spherical-Earth geometry."""
    s = torch.sin(elev_rad)
    return torch.sqrt((R_E * s) ** 2 + 2 * R_E * H_LEO + H_LEO ** 2) - R_E * s


class JointScenario:
    def __init__(self, n_users=20, n_uav=3, n_slots=12, n_leo=4, area=3000.0,
                 alt=100.0, cov_radius=900.0, v_max=140.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.K, self.M, self.N, self.Ln = n_users, n_uav, n_slots, n_leo
        self.area, self.alt = area, alt
        self.cov_radius, self.v_max = cov_radius, v_max

        # ---- ground users (z = 0) and UAV start positions (fixed altitude) ----
        xy = (torch.rand(n_users, 2, generator=g) - 0.5) * 2 * area
        self.users = torch.cat([xy, torch.zeros(n_users, 1)], dim=-1)        # (K,3)
        sx = (torch.rand(n_uav, 2, generator=g) - 0.5) * area
        self.start = torch.cat([sx, torch.full((n_uav, 1), alt)], dim=-1)    # (M,3)

        # ---- no-fly sphere near center ----
        self.nofly_c = torch.tensor([[0.0, 0.0, alt]])
        self.nofly_r = torch.tensor([400.0])
        # push any UAV that starts inside (margin-inflated) no-fly radially outward
        keep = 1.3 * float(self.nofly_r)
        rel = self.start[:, :2] - self.nofly_c[:, :2]
        rn = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
        push = (rn < keep).float()
        self.start[:, :2] = self.nofly_c[:, :2] + rel / rn * (push * keep + (1 - push) * rn)

        # ---- LEO passes: staggered raised-cosine elevation bumps that tile all N slots
        # so at least one satellite is always visible, while the best satellite (highest
        # elevation = shortest slant range) shifts over time, forcing handovers.
        n_idx = torch.arange(n_slots).float()
        sigma = n_slots / float(n_leo)
        elev = torch.zeros(n_leo, n_slots)
        for l in range(n_leo):
            c = n_slots * (l + 0.5) / n_leo                  # pass center slot
            elev[l] = 70.0 * torch.exp(-((n_idx - c) / sigma) ** 2)  # peak ~70 deg
        self.leo_elev = torch.deg2rad(elev)                                  # (Ln,N)
        self.leo_az = torch.linspace(0, 2 * math.pi, n_leo + 1)[:n_leo]      # (Ln,)
        self.min_elev = math.radians(10.0)

        # ---- radio parameters ----
        self.f_access = 2.0e9        # S-band access
        self.f_bh = 20.0e9           # Ka-band backhaul
        self.bw_access = 20e6
        self.bw_bh = 100e6
        self.ptx_access_dbm = 30.0   # UAV access tx power
        self.ptx_bh_dbm = 33.0       # UAV->LEO tx power (+ high antenna gain folded in)
        self.gain_bh_db = 35.0       # combined antenna gain on the backhaul
        self.noise_dbm = -174 + 10 * math.log10(self.bw_access) + 7.0  # +NF
        self.noise_bh_dbm = -174 + 10 * math.log10(self.bw_bh) + 3.0
        self.w_energy = 1e-4
        self.w_handover = 0.5

        # Visibility mask (Ln, N): 1 if elevation above threshold.
        self.vis = (self.leo_elev >= self.min_elev).float()

    # ---- backhaul rate each UAV gets from each LEO, per slot (B-independent) ----
    def backhaul_rate(self):
        """Return (Ln, N) spectral-efficiency [bps/Hz] of UAV->LEO link per sat/slot.

        UAV altitude is negligible vs LEO range, so the link is modeled per (sat, slot)
        independent of UAV horizontal position (standard for LEO backhaul).
        """
        d = _slant_range(self.leo_elev)                       # (Ln,N)
        pl = _fspl_db(d, self.f_bh) - self.gain_bh_db
        rx = self.ptx_bh_dbm - pl
        snr_db = rx - self.noise_bh_dbm
        snr = 10 ** (snr_db / 10.0)
        se = torch.log2(1 + snr) * self.vis                   # gated by visibility
        return se                                              # (Ln,N)

    def evaluate(self, traj, assoc):
        """traj (B,M,N,3), assoc (B,M,N) integer LEO index in [0,Ln-1].

        Returns dict: objective (B,), feasible (B,), and component metrics.
        """
        B, dev, dt = traj.shape[0], traj.device, traj.dtype
        users = self.users.to(dev, dt)
        se_bh = self.backhaul_rate().to(dev, dt)              # (Ln,N)

        # ---- access spectral efficiency to each user from each UAV ----
        d2 = ((traj.unsqueeze(-2) - users.view(1, 1, 1, self.K, 3)) ** 2).sum(-1)
        dist = torch.sqrt(d2 + 1.0)                           # (B,M,N,K)
        pl = _fspl_db(dist, self.f_access)
        rx = self.ptx_access_dbm - pl
        snr = 10 ** ((rx - self.noise_dbm) / 10.0)
        se_acc = torch.log2(1 + snr)                          # (B,M,N,K)
        in_cov = dist <= self.cov_radius
        se_acc = torch.where(in_cov, se_acc, torch.zeros_like(se_acc))

        # ---- backhaul cap per UAV per slot from its associated LEO ----
        vis = self.vis.to(dev, dt)                            # (Ln,N)
        a = assoc.clamp(0, self.Ln - 1).long()                # (B,M,N)
        # gather backhaul SE and visibility for the chosen sat at each (b,m,n)
        idx = a
        se_bh_t = se_bh.t().unsqueeze(0).unsqueeze(0)          # (1,1,N,Ln)
        se_bh_sel = torch.gather(se_bh_t.expand(B, self.M, self.N, self.Ln), -1,
                                 idx.unsqueeze(-1)).squeeze(-1)   # (B,M,N)
        vis_t = vis.t().unsqueeze(0).unsqueeze(0)
        vis_sel = torch.gather(vis_t.expand(B, self.M, self.N, self.Ln), -1,
                               idx.unsqueeze(-1)).squeeze(-1)     # (B,M,N)
        cap = self.bw_bh / self.bw_access * se_bh_sel * vis_sel    # access-equiv cap [bps/Hz]

        # ---- deliver access rate, throttled by backhaul cap, best UAV per user ----
        load = se_acc.sum(-1)                                 # (B,M,N) demanded by UAV
        scale = (cap / load.clamp_min(1e-6)).clamp(max=1.0)
        se_acc = se_acc * scale.unsqueeze(-1)
        best = se_acc.max(dim=1).values                       # (B,N,K) best UAV per user
        throughput = best.sum(dim=(-1, -2)) * self.bw_access / 1e6   # [Mbps]

        # ---- energy + handover ----
        steps = traj[:, :, 1:, :] - traj[:, :, :-1, :]
        path = torch.linalg.norm(steps[..., :2], dim=-1)
        energy = path.sum(dim=(-1, -2))
        handover = (a[:, :, 1:] != a[:, :, :-1]).float().sum(dim=(-1, -2))

        objective = throughput - self.w_energy * energy - self.w_handover * handover

        # ---- feasibility ----
        speed_ok = (path <= self.v_max + 1e-3).all(dim=(-1, -2))
        nf = 0.5 * (self.nofly_r.to(dev, dt) ** 2
                    - ((traj - self.nofly_c.to(dev, dt).view(1, 1, 1, 3)) ** 2).sum(-1))
        nofly_ok = (nf <= 0).all(dim=(-1, -2))
        vis_ok = (vis_sel > 0).all(dim=(-1, -2))              # never associate invisible LEO
        feasible = speed_ok & nofly_ok & vis_ok

        return {
            "objective": objective, "feasible": feasible,
            "throughput": throughput, "energy": energy, "handover": handover,
            "speed_ok": speed_ok, "nofly_ok": nofly_ok, "vis_ok": vis_ok,
        }
