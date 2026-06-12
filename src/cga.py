"""Sparse Conformal Geometric Algebra (CGA) engine for Cl(4,1), PyTorch-batched.

We do NOT materialize the full 32-dimensional multivector algebra. Instead we use
the standard CGA linearization: IPNS objects (points, spheres, planes) are grade-1
vectors of the 5D Minkowski space R^{4,1}, and rigid motions SE(3) act on them as
5x5 matrices (a subgroup of the conformal group). All geometric constraints reduce
to inner products with the conformal metric eta. This keeps everything batchable on
CUDA and avoids the cost of the full geometric product.

Basis order: [e1, e2, e3, e0, einf]
Metric eta: e_i.e_j = delta_ij (i,j in 1..3), e0.e0 = einf.einf = 0, e0.einf = -1.
"""

import torch

# Conformal metric in basis [e1, e2, e3, e0, einf].
# <a, b> = a^T ETA b.
ETA = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, -1.0, 0.0],
    ]
)


def inner(a, b):
    """Conformal inner product <a, b> over the last dim (size 5). Broadcasts."""
    eta = ETA.to(a.device, a.dtype)
    return torch.einsum("...i,ij,...j->...", a, eta, b)


def point(x):
    """Embed Euclidean points x (..., 3) as conformal points P (..., 5).

    P = x + e0 + 0.5|x|^2 einf. Null vectors: <P, P> = 0.
    """
    x2 = (x * x).sum(-1, keepdim=True)  # (..., 1)
    e0 = torch.ones_like(x2)
    einf = 0.5 * x2
    return torch.cat([x, e0, einf], dim=-1)


def euclid(P):
    """Recover Euclidean coordinates (..., 3) from a conformal point (..., 5).

    A finite conformal point has nonzero e0 component; we normalize by it.
    """
    e0 = P[..., 3:4]
    return P[..., 0:3] / e0


def sphere(center, radius):
    """IPNS sphere S = C - 0.5 r^2 einf, where C = point(center). (..., 5).

    Test: <P, S> = 0.5 (r^2 - |x - c|^2). So <P, S> >= 0 iff x is inside the sphere.
    """
    C = point(center)
    r2 = (radius ** 2).unsqueeze(-1) if radius.dim() == center.dim() - 1 else radius ** 2
    out = C.clone()
    out[..., 4] = out[..., 4] - 0.5 * r2.squeeze(-1)
    return out


def plane(normal, offset):
    """IPNS plane pi = n + d einf, n unit normal (..., 3), d offset scalar (...,).

    Test: <P, pi> = x.n - d = signed distance to the plane. Half-space via sign.
    """
    z = torch.zeros(normal.shape[:-1] + (1,), device=normal.device, dtype=normal.dtype)
    d = offset.unsqueeze(-1) if offset.dim() == normal.dim() - 1 else offset
    return torch.cat([normal, z, d], dim=-1)


def _skew(w):
    """Map axis-angle vectors w (..., 3) to skew-symmetric matrices (..., 3, 3)."""
    O = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    O[..., 0, 1] = -w[..., 2]
    O[..., 0, 2] = w[..., 1]
    O[..., 1, 0] = w[..., 2]
    O[..., 1, 2] = -w[..., 0]
    O[..., 2, 0] = -w[..., 1]
    O[..., 2, 1] = w[..., 0]
    return O


def rotation3(w):
    """Rodrigues: axis-angle w (..., 3) to SO(3) matrix (..., 3, 3)."""
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-12)  # (..., 1)
    K = _skew(w / theta)
    th = theta.unsqueeze(-1)  # (..., 1, 1)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(w.shape[:-1] + (3, 3))
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def motor(w, t):
    """Build the 5x5 conformal matrix of the rigid motion (rotate by w, then translate t).

    w: axis-angle (..., 3), t: translation (..., 3). Returns M (..., 5, 5) acting on
    grade-1 conformal vectors as v' = M v. Equivalent to the CGA versor sandwich
    M v M~ restricted to grade 1, but computed as a plain matrix for speed.
    """
    batch = w.shape[:-1]
    dev, dt = w.device, w.dtype
    R = rotation3(w)  # (..., 3, 3)

    # Translation matrix L_t in basis [e1,e2,e3,e0,einf]; columns = images of basis.
    L = torch.eye(5, device=dev, dtype=dt).expand(batch + (5, 5)).clone()
    L[..., 4, 0] = t[..., 0]
    L[..., 4, 1] = t[..., 1]
    L[..., 4, 2] = t[..., 2]
    L[..., 0, 3] = t[..., 0]
    L[..., 1, 3] = t[..., 1]
    L[..., 2, 3] = t[..., 2]
    L[..., 4, 3] = 0.5 * (t * t).sum(-1)

    # Rotation embedded as 5x5 (identity on e0, einf).
    Rb = torch.eye(5, device=dev, dtype=dt).expand(batch + (5, 5)).clone()
    Rb[..., 0:3, 0:3] = R

    return L @ Rb  # rotate first, then translate


def apply_motor(M, v):
    """Apply motor matrix M (..., 5, 5) to conformal vector(s) v (..., 5) -> (..., 5)."""
    return torch.einsum("...ij,...j->...i", M, v)
