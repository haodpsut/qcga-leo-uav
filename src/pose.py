"""Full-SE(3)-pose coverage and decoders for the warm-start (B2) study.

The UAV beam footprint is ELLIPTICAL, so the full orientation of the UAV (including
roll about the boresight) matters, not just the pointing direction. This is the regime
where a minimal Euclidean attitude parameterization (Euler angles) suffers gimbal lock
while the CGA rotor / bivector parameterization stays singularity-free.

Two attitude parameterizations, identical in every other respect:
  cga_fullpose_decode    -> attitude as accumulated rotors (bivector composition)
  euclid_fullpose_decode -> attitude as Euler angles (Z-Y-X)

The point of B2: between epochs the scene undergoes a KNOWN 3D rotation g. Warm-starting
the next epoch means transforming the previous solution by g. The CGA solution composes
the rotor g exactly; the Euler solution must re-extract angles from g @ R, which is
ill-conditioned near pitch = +/- 90 deg.
"""

import torch

import cga


def euler_to_R(angles):
    """Z-Y-X intrinsic Euler angles (..., 3) -> rotation matrix (..., 3, 3)."""
    z, y, x = angles[..., 0], angles[..., 1], angles[..., 2]
    cz, sz = torch.cos(z), torch.sin(z)
    cy, sy = torch.cos(y), torch.sin(y)
    cx, sx = torch.cos(x), torch.sin(x)
    R = torch.zeros(angles.shape[:-1] + (3, 3), device=angles.device, dtype=angles.dtype)
    R[..., 0, 0] = cz * cy
    R[..., 0, 1] = cz * sy * sx - sz * cx
    R[..., 0, 2] = cz * sy * cx + sz * sx
    R[..., 1, 0] = sz * cy
    R[..., 1, 1] = sz * sy * sx + cz * cx
    R[..., 1, 2] = sz * sy * cx - cz * sx
    R[..., 2, 0] = -sy
    R[..., 2, 1] = cy * sx
    R[..., 2, 2] = cy * cx
    return R


def R_to_euler(R):
    """Rotation matrix (..., 3, 3) -> Z-Y-X Euler angles (..., 3).

    Singular at pitch = +/- 90 deg (R[2,0] = -/+1): yaw and roll collapse onto one axis
    and cannot be separated. We clamp and zero the roll there, which is exactly the
    information loss that makes Euler warm-starts brittle.
    """
    sy = -R[..., 2, 0]
    sy = sy.clamp(-1.0, 1.0)
    y = torch.asin(sy)
    cy = torch.cos(y)
    near = cy.abs() < 1e-6
    z = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    x = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    # Gimbal lock fallback: fold yaw+roll, drop the unrecoverable component.
    z_s = torch.atan2(-R[..., 0, 1], R[..., 1, 1])
    z = torch.where(near, z_s, z)
    x = torch.where(near, torch.zeros_like(x), x)
    return torch.stack([z, y, x], dim=-1)


def _bounded_positions(p_xyz, scn):
    """Shared bounded-increment kinematics; p_xyz (B,M,N,3) increments -> positions."""
    B, M, N = p_xyz.shape[0], scn.M, scn.N
    dev, dt = p_xyz.device, p_xyz.dtype
    dz_max = 30.0
    pos = torch.zeros(B, M, N, 3, device=dev, dtype=dt)
    pos[:, :, 0, :] = scn.start.to(dev, dt).view(1, M, 3)
    for n in range(1, N):
        disp = p_xyz[:, :, n, :]
        horiz = disp[..., :2]
        hn = torch.linalg.norm(horiz, dim=-1, keepdim=True).clamp_min(1e-9)
        horiz = horiz * (scn.v_max * torch.tanh(hn / scn.v_max) / hn)
        dz = dz_max * torch.tanh(disp[..., 2:3] / dz_max)
        pos[:, :, n, :] = pos[:, :, n - 1, :] + torch.cat([horiz, dz], dim=-1)
    return pos


def cga_fullpose_decode(params, scn, Rleft=None):
    """params (B, M*N*6) -> (pos (B,M,N,3), R (B,M,N,3,3)) via composed rotors.

    Rleft (B,M,3,3) optional world-frame rotation applied on the LEFT of every attitude
    (R_n <- Rleft @ R_n). A scene rotation g acts on body->world poses as g @ R, so a
    warm start across a known rotation g passes Rleft = g and leaves the per-slot
    bivector turns untouched: the rotor carries g exactly and singularity-free.
    """
    B, M, N = params.shape[0], scn.M, scn.N
    dev, dt = params.device, params.dtype
    p = params.view(B, M, N, 6)
    pos = _bounded_positions(p[..., 3:6], scn)  # world-frame position-increment channel

    R_acc = torch.eye(3, device=dev, dtype=dt).expand(B, M, 3, 3).contiguous()
    R = torch.zeros(B, M, N, 3, 3, device=dev, dtype=dt)
    R[:, :, 0] = R_acc
    turn_max = 0.6
    for n in range(1, N):
        w = p[:, :, n, 0:3]
        wn = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
        w = w * (turn_max * torch.tanh(wn / turn_max) / wn)
        R_acc = cga.rotation3(w) @ R_acc
        R[:, :, n] = R_acc
    if Rleft is not None:
        Rl = Rleft.to(dev, dt).reshape(Rleft.shape[0], M, 1, 3, 3)  # broadcast over B, N
        R = Rl @ R
    return pos, R


def euclid_fullpose_decode(params, scn, Rleft=None):
    """params (B, M*N*6) -> (pos, R) with attitude as absolute Euler angles per slot.

    Rleft optional world-frame left rotation (R_n <- Rleft @ euler(theta_n)). The fair
    Euler baseline does NOT use it for warm starts: an Euler-parameterized solver has to
    fold the known rotation g back into its angle variables via R_to_euler(g @ R), which
    is ill-conditioned near pitch = 90 deg. Provided only for symmetry of the interface.
    """
    B, M, N = params.shape[0], scn.M, scn.N
    dev, dt = params.device, params.dtype
    p = params.view(B, M, N, 6)
    pos = _bounded_positions(p[..., 0:3], scn)
    # Map the raw attitude channel to bounded Euler angles in roughly (-pi, pi) so the
    # optimizer searches a single wrap, not the ~24-turn space the raw [-150,150] range
    # would otherwise impose. This keeps the Euler baseline FAIR vs the rotor turns.
    angles = torch.pi * torch.tanh(p[..., 3:6] / 80.0)
    Rloc = euler_to_R(angles)  # (B,M,N,3,3)
    if Rleft is not None:
        Rl = Rleft.to(dev, dt).reshape(Rleft.shape[0], M, 1, 3, 3)
        Rloc = Rl @ Rloc
    return pos, Rloc


def elliptical_quality(scn, pos, R, ang_y=0.9, ang_z=0.35):
    """Objective with an elliptical beam: orientation (incl. roll) matters.

    A user is served by UAV (m,n) if it is in range AND its direction, expressed in the
    UAV body frame, lies inside an ellipse with half-angles (ang_y, ang_z) about the
    boresight (body +x). Because ang_y != ang_z the roll of R changes who is covered.
    Returns objective (B,).
    """
    dev, dt = pos.device, pos.dtype
    users = scn.users.to(dev, dt)
    rel = users.view(1, 1, 1, scn.K, 3) - pos.unsqueeze(-2)          # (B,M,N,K,3)
    d = torch.linalg.norm(rel, dim=-1).clamp_min(1e-9)
    rel_u = rel / d.unsqueeze(-1)
    # Express direction in body frame: body = R^T world.
    relb = torch.einsum("bmnji,bmnkj->bmnki", R, rel_u)             # (B,M,N,K,3)
    fwd = relb[..., 0].clamp_min(1e-6)
    ay = torch.atan2(relb[..., 1], fwd)
    az = torch.atan2(relb[..., 2], fwd)
    in_fov = (relb[..., 0] > 0) & ((ay / ang_y) ** 2 + (az / ang_z) ** 2 <= 1.0)
    in_range = d <= scn.cov_radius
    inside = in_fov & in_range

    snr = 1e6 / (d ** 2 + 1.0)
    rate = torch.where(inside, torch.log2(1.0 + snr), torch.zeros_like(d))
    best = rate.max(dim=1).values            # best UAV per (slot,user)
    access = best.sum(dim=(-1, -2))
    steps = pos[:, :, 1:, :] - pos[:, :, :-1, :]
    energy = torch.linalg.norm(steps[..., :2], dim=-1).sum(dim=(-1, -2))
    return access - 0.002 * energy
