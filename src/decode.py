"""Two ways to turn an optimizer parameter vector into a UAV trajectory.

cga_decode:   bounded body-frame increments composed as motors. Speed and turn-rate
              limits are enforced by construction, so every decoded trajectory is
              kinematically feasible. This is the native-feasibility property of the
              CGA / motor parameterization.

euclid_decode: raw absolute waypoints in R^3. Kinematic limits are NOT guaranteed;
              violations must be pushed out by a penalty during optimization.

Both share the start pose and the no-fly constraint (neither parameterization makes
the no-fly region automatically satisfied), so the comparison isolates the kinematic
feasibility advantage of the motor parameterization.
"""

import torch

import cga


def cga_decode(params, scn):
    """params (B, M*N*6) -> traj (B, M, N, 3), all kinematically feasible.

    Per slot: (w in R^3 turn, t_body in R^3 body-frame translation). Heading rotation
    accumulates via Rodrigues (engine cga.rotation3); the body translation is rotated
    into the world frame, then clamped to the speed / climb limits.
    """
    B = params.shape[0]
    M, N = scn.M, scn.N
    dev, dt = params.device, params.dtype
    p = params.view(B, M, N, 6)

    start = scn.start.to(dev, dt)                  # (M, 3)
    pos = torch.zeros(B, M, N, 3, device=dev, dtype=dt)
    pos[:, :, 0, :] = start.view(1, M, 3)

    R_acc = torch.eye(3, device=dev, dtype=dt).expand(B, M, 3, 3).contiguous()
    turn_max = 0.6  # rad per slot
    dz_max = 30.0   # vertical step [m]

    for n in range(1, N):
        w = p[:, :, n, 0:3]
        wn = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
        w = w * (turn_max * torch.tanh(wn / turn_max) / wn)   # smooth clamp |w|<=turn_max
        R_acc = cga.rotation3(w) @ R_acc
        t_body = p[:, :, n, 3:6]
        disp = torch.einsum("bmij,bmj->bmi", R_acc, t_body)
        horiz = disp[..., :2]
        hn = torch.linalg.norm(horiz, dim=-1, keepdim=True).clamp_min(1e-9)
        horiz = horiz * (scn.v_max * torch.tanh(hn / scn.v_max) / hn)  # |horiz|<=v_max
        dz = dz_max * torch.tanh(disp[..., 2:3] / dz_max)
        pos[:, :, n, :] = pos[:, :, n - 1, :] + torch.cat([horiz, dz], dim=-1)
    return pos


def euclid_decode(params, scn):
    """Absolute-waypoint Euclidean decode (STRAWMAN, kept only for reference).

    params (B, M*N*3) -> traj (B, M, N, 3). Slot 0 forced to the start pose. Speed is
    not constrained, so the swarm is almost never feasible; do not use as the headline
    baseline. Use euclid_incr_decode for the fair comparison.
    """
    B = params.shape[0]
    M, N = scn.M, scn.N
    dev, dt = params.device, params.dtype
    traj = params.view(B, M, N, 3).clone()
    traj[:, :, 0, :] = scn.start.to(dev, dt).view(1, M, 3)
    return traj


def euclid_incr_decode(params, scn):
    """Fair Euclidean baseline: bounded world-frame increments (no motor structure).

    params (B, M*N*3) -> traj (B, M, N, 3). Each slot adds a world-frame displacement
    whose horizontal norm is clamped to v_max and vertical step to dz_max, exactly the
    same kinematic envelope as cga_decode. The ONLY difference from cga_decode is the
    parameterization: plain Cartesian increments here vs composed SE(3) motors there.
    This isolates the effect of the CGA geometry (convergence, quality, equivariance)
    rather than a trivial feasibility gap.
    """
    B = params.shape[0]
    M, N = scn.M, scn.N
    dev, dt = params.device, params.dtype
    p = params.view(B, M, N, 3)
    dz_max = 30.0

    pos = torch.zeros(B, M, N, 3, device=dev, dtype=dt)
    pos[:, :, 0, :] = scn.start.to(dev, dt).view(1, M, 3)
    for n in range(1, N):
        disp = p[:, :, n, :]
        horiz = disp[..., :2]
        hn = torch.linalg.norm(horiz, dim=-1, keepdim=True).clamp_min(1e-9)
        horiz = horiz * (scn.v_max * torch.tanh(hn / scn.v_max) / hn)
        dz = dz_max * torch.tanh(disp[..., 2:3] / dz_max)
        pos[:, :, n, :] = pos[:, :, n - 1, :] + torch.cat([horiz, dz], dim=-1)
    return pos


def fitness(traj, scn, feas_penalty=50.0):
    """Penalized fitness for the optimizer + raw metrics for reporting.

    Returns (fit (B,), info dict). The penalty pushes the optimizer toward feasible
    trajectories without hard-rejecting them.
    """
    out = scn.evaluate(traj)
    # Soft penalties mirror the hard feasibility flags reported in `out`.
    steps = traj[:, :, 1:, :] - traj[:, :, :-1, :]
    path = torch.linalg.norm(steps[..., :2], dim=-1)
    speed_viol = (path - scn.v_max).clamp_min(0.0).sum(dim=(-1, -2))
    Pm = cga.point(traj)
    nf = cga.inner(Pm, scn.nofly.to(traj.device, traj.dtype).view(1, 1, 1, 5))
    nofly_viol = nf.clamp_min(0.0).sum(dim=(-1, -2))
    fit = out["objective"] - feas_penalty * (speed_viol / scn.v_max + nofly_viol / 1e5)
    return fit, out
