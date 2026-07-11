# eshkol_audit — Prestige Prompt (Jul 11 2026 v1)

## Purpose
Fork of `tsotchke/eshkol` for independent, reproducible cross-validation of its
geometry math against `waefrebeorn/WuBuMath`. Inviting to upstream, rigorous in method.

## One-liner Mission
Prove or refute eshkol's geometry claims with numerical + symbolic checks, and
show tsotchke how a Lean+C-contract pipeline makes "quantum/proven" claims falsifiable.

## Stack
- Upstream: Scheme-dialect compiler (C++ frontend/LLVM codegen, C VM) + `lib/core/manifold.esk` (pure-Scheme geometry).
- Audit tooling: standalone C (`cross-validation/`), no LLVM needed.
- Reference truth: `waefrebeorn/WuBuMath` (`wubu_poincare_geom.c`, `lean/WubuProofs/`).

## Priority Queue
- P0 — keep `manifold.esk` verbatim; document the conformal-factor bug (done, reproducible).
- P0 — upstream PR: fix `manifold-exp-map` (conformal factor scales vector, not inside tanh).
- P1 — extend `cross-validation/` to sphere + product manifolds.
- P1 — port our `FiberBundle.lean` SO(3) proof to cross-check eshkol's rotation ops.
- P2 — document the "consciousness engine"/HoTT claims as unproven (need LLVM-21 build).

## Key Math (verified)
- Poincaré exp (origin): `exp_0(v) = tanh(|v|)/|v| · v`; `dist(0,exp_0(v)) = 2|v|`.
- Christoffel (Poincaré): `Γⁱ_jk = c·λ·(δⁱ_k x_j + δⁱ_j x_k − δ_jk xⁱ)`, `λ = 2/(1−c|x|²)`.
- Curvature: `K = −c` (Poincaré), `+c` (sphere); `R = K·n(n−1)`.
- eshkel analytic Christoffel (manifold_christoffel) is CORRECT; its exp-map is NOT.

## Verified (TRUST: HIGH)
- eshkol `manifold-exp-map` violates geodesic invariant for p≠0 (reproduced in `cross-validation/`).
- WuBuMath `exp_0`+distance self-consistent.
- Analytic curvature/K/Ricci values correct.

## Debatable (verify before claiming)
- Whether eshkol's exp/distance "convention drift" is intended (header says dylib is source of truth).
- "Consciousness engine" / HoTT foundations — documentation intent, unexercised.

## Data Not To Re-derive
- eshkol needs LLVM 21 (absent here) → cannot build REPL; audit math via standalone C.
- `vm_geometric.c` does NOT exist in git tree; geometric math is only in `manifold.esk`.

## Full Context
Read `.hermes/mind-palace/goal-mantra.md` for orientation; `plans/devils_advocate_v1.md` for the audit; `ESHKOL_STRUCTURE.md` for architecture.
