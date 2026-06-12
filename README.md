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

## Status / TODO

- [x] Sparse CGA engine + scenario + QPSO + CGA-vs-Euclidean smoke test.
- [ ] **Fair Euclidean baseline**: bounded incremental waypoints (not absolute) so
      the comparison isolates convergence/quality, not a trivial feasibility win.
- [ ] 3GPP TR 38.811 NTN path loss (UAV-LEO) + air-to-ground access model.
- [ ] Multi-LEO association + handover penalty + backhaul capacity.
- [ ] SE(3)-equivariance theorem + invariance experiment.
- [ ] Rigorous baselines: well-tuned GA (genetic), PSO, DE, CMA-ES, L-SHADE, SCA.
- [ ] Prior-art / novelty deep-research pass before writing related work.

Note: in this work "GA" always means **Geometric Algebra**; the genetic-algorithm
baseline is written out as "Genetic Algorithm".
