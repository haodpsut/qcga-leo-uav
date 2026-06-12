# Quantum-Inspired Optimization in Conformal Geometric Algebra for Joint UAV Trajectory and LEO Association

Method paper (IEEE Transactions target) for an integrated LEO-UAV-ground (SAGIN)
network. A quantum-behaved swarm optimizer searches over **motors** (rigid-motion
versors) in Conformal Geometric Algebra so that geometric constraints (coverage,
no-fly, line-of-sight to LEO) become inner-product tests and UAV kinematics are
feasible by construction.

## Why CGA

In CGA (`Cl(4,1)`) the points, coverage spheres, no-fly regions, and LEO visibility
half-spaces are all grade-1 vectors, and every constraint reduces to a conformal
inner product. Rigid motions act linearly as 5x5 matrices, so the optimizer searches
the minimal 6-DoF SE(3) parameterization directly. The conjectured wins, each with a
dedicated experiment:

1. **Feasibility** — bounded motors keep trajectories kinematically valid by design.
2. **Convergence** — the rotor manifold gives a smoother landscape.
3. **Solution quality** — better objective at equal fitness-evaluation budget.
4. **Equivariance** — SE(3)-equivariant search; performance invariant to global
   rotation of the scene where Euclidean baselines degrade.

## Layout

    src/cga.py        sparse CGA engine (5D IPNS + 5x5 motors), PyTorch-batched
    src/scenario.py   LEO-UAV-ground geometry, channel proxy, objective + constraints
    src/decode.py     CGA-motor decode vs Euclidean-waypoint decode + fitness
    src/qpso.py       quantum-behaved PSO (shared optimizer for the ablation)
    experiments/      runnable studies (start: smoke_test.py)

## Run

    conda env create -f environment.yml
    conda activate qcga-leo-uav
    python experiments/smoke_test.py

## VERDICT (2026-06-12): parked — no Transactions-grade quantitative contribution

After exhaustive honest testing, this direction does not yield a defensible method-paper
result: (1) CGA is notation only (= quaternion pose, = vector-algebra constraints to
1e-11); (2) singularity-free attitude is known since 1982/2005 and helps only in steep
nadir/zenith edge cases; (3) on the full joint LEO-UAV problem the quantum-inspired QPSO
ties or loses to GWO/DE/GA with no robust win across scales (incumbent GWO best at the
largest scale); (4) the 3-way combination is novel only as framing. Prior-art deep
research confirmed each pillar is occupied. Decision: stop; redirect effort to stronger
portfolio papers. Codebase (CGA engine, joint scenario, 5 solvers, 6 experiments) is
reusable. Full reasoning trail below.

## Findings so far (honest)

1. Position-only trajectory (smoke): with a FAIR Euclidean baseline, CGA gives no
   advantage (extra rotational DoF are dead weight). `smoke_test.py`.
2. Equivariance under scene rotation: CGA only marginally more rotation-stable, not
   better on quality. `equivariance_test.py`.
3. Warm-start across rotation: retired. The problem's only symmetry is yaw (anisotropic
   UAV kinematics), where Euler is also singularity-free, so no gap to exploit.
   `warmstart_test.py` (superseded).
4. **The real win.** When good solutions need steep near-nadir / near-zenith pointing
   (UAV serving ground users, or tracking an overhead LEO) the singularity-free rotor
   pose parameterization beats minimal Euler angles by ~50% in coverage objective and
   converges faster, at equal budget and fair angle scaling. `attitude_param_test.py`:
   CGA 135.5 +/- 8.6 vs Euler 89.0 +/- 7.4 (5 seeds). NOTE: quaternions share this
   benefit (unit quaternion = rotor); the CGA contribution is UNIFYING this with the
   conformal constraint algebra and motions, not the gimbal-lock observation itself.

## TODO

- [x] Fair Euclidean baseline; fair Euler angle scaling.
- [x] Locate a genuine, mechanism-backed CGA advantage (steep-pointing attitude search).
- [ ] Reframe paper around the honest steep-pointing result (see strategy notes).
- [ ] 3GPP TR 38.811 NTN path loss (UAV-LEO) + air-to-ground access model.
- [ ] Multi-LEO association + handover penalty + backhaul capacity.
- [ ] Quaternion baseline (to show CGA matches it) + show CGA's added constraint-algebra value.
- [ ] Rigorous baselines: well-tuned Genetic Algorithm, PSO, DE, CMA-ES, L-SHADE, SCA.
- [ ] Prior-art / novelty deep-research pass before writing related work.

Note: in this work "GA" always means **Geometric Algebra**; the genetic-algorithm
baseline is written out as "Genetic Algorithm".
