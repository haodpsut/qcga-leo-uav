# Formulation note — joint LEO-UAV trajectory + association (Transactions-grade)

Status: framing-independent core (system model, channel, MINLP, CGA encoding,
decomposition, equivariance theorem, experiment/ablation matrix). The novelty framing
and the choice of "hero" baseline are pending the prior-art deep-research verdict.

Notation: bold lowercase = vectors, bold uppercase = matrices/multivectors, `[n]` =
discrete time slot. SI units.

---

## 1. System model

### 1.1 Sets and indices
- Ground users / IoT devices: `k in K = {1..K}`, fixed positions `g_k in R^3`, `z=0`.
- UAV aerial base stations: `m in M = {1..M}`, pose per slot.
- LEO satellites: `l in L`, ephemeris-known positions `s_l[n]` (TLE / circular-orbit).
- Time horizon `T` split into `N` slots of length `dt`; slot index `n in {0..N-1}`.

### 1.2 Decision variables
- UAV position `q_m[n] in R^3` (or as a conformal point, Sec. 4).
- UAV attitude `R_m[n] in SO(3)` (boresight + roll of an elliptical beam).
- UAV-LEO association `a_m[n] in L_vis(m,n) ∪ {0}` (0 = no backhaul this slot).
- (Optional v2) access power `p_m[n]`, bandwidth share.

### 1.3 Channel model (3GPP TR 38.811 NTN + air-to-ground)
- UAV-LEO backhaul: TR 38.811 NTN path loss at the chosen band (S / Ka), free-space +
  atmospheric + shadow-fading term; elevation-angle-dependent. Visibility `L_vis(m,n)`
  by min elevation `theta_min`.
- UAV-user access: air-to-ground LoS/NLoS model (e.g., 3GPP UAV / elevation-prob LoS).
- SNR_access(k,m,n), backhaul capacity `C_bh(m,n)` from the associated LEO link.

### 1.4 Objective
Maximize delivered sum rate minus propulsion energy and handover churn:

    U = sum_n sum_k  r_k[n]  -  w_e * E_prop  -  w_h * H_assoc

- `r_k[n]` = rate of user k = served by the best covering UAV, throttled by that UAV's
  backhaul capacity (access-backhaul coupling).
- `E_prop` = rotary-wing propulsion energy (Zeng 2019 standard model) over the path.
- `H_assoc` = number of association changes `1[a_m[n] != a_m[n-1]]` (handover penalty).

---

## 2. Constraints (standard form; CGA inner-product form in Sec. 4)

1. Access coverage: user k served by (m,n) only if inside the elliptical beam AND in
   range: `||g_k - q_m[n]|| <= R_cov` and angular test in the body frame of `R_m[n]`.
2. UAV kinematics (anisotropic): horizontal step `||Δq_xy|| <= v_max*dt`, climb
   `|Δq_z| <= w_max*dt`, turn rate `||ω|| <= ω_max*dt`. (Anisotropy ⇒ the symmetry
   group is yaw + translation, NOT full SO(3); see equivariance theorem caveat.)
3. No-fly zones: `q_m[n]` outside given spheres/cylinders.
4. Inter-UAV separation: `||q_m[n] - q_m'[n]|| >= d_min`.
5. LEO visibility / LOS: elevation to `a_m[n]` above `theta_min` (half-space test).
6. Backhaul capacity: sum of access rate routed by UAV m <= `C_bh(m,n)`.

This is a mixed-integer nonlinear program (MINLP): continuous pose, integer
association; nonconvex coverage/channel ⇒ NP-hard ⇒ motivates the metaheuristic.

---

## 3. Problem decomposition

The joint MINLP separates into two coupled subproblems:

- P1 (continuous): trajectory + attitude `{q_m[n], R_m[n]}` for fixed association.
- P2 (combinatorial): association `{a_m[n]}` for fixed trajectory (assignment under
  visibility + backhaul capacity + handover penalty).

Two solve modes to evaluate:
- (D1) Alternating optimization: solve P1 with QI optimizer, then P2 by capacitated
  assignment / qubit-measurement (QIO-style), iterate to fixed point.
- (D2) Joint encoding: one particle carries pose params + a quantum-bit association
  vector; QPSO + measurement optimize them together.

Complexity: per fitness eval O(M N (K + |L_vis|)); per QPSO iter O(P · that), P =
swarm. Report wall-clock vs scale (M, K, N) and vs each baseline.

---

## 4. CGA encoding (Cl(4,1), basis [e1,e2,e3,e0,einf])

Honest scope (established empirically, see repo experiments): the CGA layer is a
UNIFYING formalism; its constraint inner products equal plain vector algebra to machine
precision, and its rotor pose equals quaternion pose. We present it as a single
algebraic language for points, regions, motions, and a singularity-free pose, not as a
source of a different/cheaper computation.

- Conformal point: `P = x + e0 + 0.5|x|^2 einf`, null `<P,P> = 0`, and
  `<P1,P2> = -0.5|x1-x2|^2` (squared distance for free).
- Coverage sphere: `S = C - 0.5 R_cov^2 einf`; `<P_k, S> >= 0` ⇔ user inside range.
- No-fly sphere `N_j`: `<P_m, N_j> <= 0` (outside).
- LOS / elevation half-space: plane `pi = n + d einf`; `<P_m, pi>` = signed distance.
- Separation: `<P_m, P_m'> <= -0.5 d_min^2`.
- Rigid motion (rotate+translate) = motor `M` (5x5 conformal matrix); `P' = M P` and
  attitude composes as rotors (singularity-free, the only measured pose advantage vs
  Euler, shared with quaternions).

---

## 5. Equivariance theorem (statement)

Let `g` be a yaw rotation + translation (the symmetry group respected by the anisotropic
kinematics). Claim: the objective `U` is invariant under applying `g` jointly to all
users, UAV poses, and LEO geometry; and the CGA motor parameterization is equivariant,
so if `X*` solves scenario `Σ` then `g·X*` solves `g·Σ`. HONEST CAVEAT to prove and
state: this holds for yaw, NOT pitch/roll (anisotropic speed/climb limits break full
SO(3)); under yaw the Euler yaw angle is already singularity-free, so equivariance is a
structural/clarity property here, not a performance lever. Report it as such.

---

## 6. Experiment + ablation matrix (Transactions-grade)

Protocol: 30 seeds per config; mean ± std; Wilcoxon signed-rank vs runner-up; fixed
seeds, conda env, public repo. Channel = TR 38.811 + A2G.

### 6.1 Main comparison (objective, feasibility, runtime, convergence)
Baselines, all fairly tuned:
- Proposed QI-CGA (rotor pose + conformal constraints + QPSO, modes D1 and D2).
- Genetic Algorithm (proportional pop + adaptive mutation), PSO, DE, CMA-ES, L-SHADE.
- SCA / convex-approximation trajectory (classical UAV-traj standard).
- Association baselines: nearest-LEO, max-elevation, greedy-capacitated.

### 6.2 Ablations (isolate each design choice)
- A. Pose parameterization: rotor (CGA) vs quaternion vs Euler. Expectation:
  rotor ≈ quaternion >> Euler, and ONLY in steep nadir/zenith regimes (sweep the
  pointing-steepness / user-altitude geometry to show where the gap appears/vanishes).
- B. Constraint layer: CGA inner product vs plain vector algebra — report numerical
  equivalence (≈1e-11) to be transparent that this is notation, plus any code-clarity
  or derivation benefit.
- C. Optimizer: QPSO vs PSO vs DE vs GA at equal budget (convergence + final).
- D. Solve mode: alternating (D1) vs joint quantum-bit encoding (D2).
- E. Equivariance: with vs without exploiting yaw symmetry (search-space reduction).
- F. Scale: M in {1..4}, K in {10..100}, N in {8..32}; runtime + quality scaling.
- G. Channel realism: free-space proxy vs full TR 38.811 (sensitivity).

### 6.3 Figures (≈11)
TikZ: SAGIN 3-layer system; CGA object cheatsheet; QI-CGA pipeline; decomposition /
equivariance diagram. Results: convergence; feasibility-rate; pose-param gap vs
pointing-steepness (the honest hero plot); objective vs M/K; energy-throughput Pareto;
runtime vs scale; Wilcoxon table; ablation bars.

---

## 7. Open decision (gates the framing)
Pending deep-research: is singularity-free / quaternion attitude in UAV-trajectory
optimization already standard? If novel-in-LEO-UAV ⇒ honest-paper framing above. If
well-trodden ⇒ pivot to CGA-equivariant learning (where GA/equivariance pays a real
data-efficiency dividend), reusing Secs. 1-4 system model and channel.
