# eshkol_audit — Triple Devil's-Advocate Audit (v1, Jul 11 2026)

Subject: `tsotchke/eshkol` `lib/core/manifold.esk` (release d861d20a)
Method: 4-phase Claim→Verify→Risk→Mitigate, run 3×. Reference truth: WuBuMath.

---

## PHASE 1 — Affirm & Steelman (strongest positive case)

eshkol ships a genuinely useful constant-curvature geometry library in pure
Scheme: exp/log maps, geodesic distance, parallel transport, AND closed-form
Christoffel / sectional / Ricci / scalar / Riemann curvature for Euclidean,
Poincaré (K=−1), and sphere (K=+1). The curvature formulas are textbook-correct.
The library runs under REPL/JIT/AOT with no native-builtin dependency. This is
real, reusable math — not vapor.

## PHASE 2 — Attack & Devil's-Advocate (strongest counter)

**Claim A:** "manifold-exp-map is a correct Riemannian exponential map."
- Verify: `exp_p(v)=p⊕(tanh(0.5·lam·|v|)/|v|·v)`, `lam=2/(1−|p|²)`.
- Attack: the geodesic invariant `dist(p,exp_p(v)) = const·|v|` must hold.
  Measured ratio `dist/|v|` drifts 2.1→3.1 as `|v|` grows (500 random bases).
  **FAILS for p≠0.** Root cause: `lam` is inside `tanh`; correct form scales the
  *vector* (`p⊕(lam·tanh(|v|/2)/|v|·v)`).
- Risk: any code using `exp_p` off the origin (geodesic shooting, parallel
  transport composition, optimization on the manifold) gets wrong distances.
- Mitigate: documented in REPORT.md §F1; keep `manifold.esk` verbatim; PR upstream.

**Claim B:** "High-performance, mathematically rigorous foundation."
- Attack: build is broken without LLVM 21 (huge footprint for a geometry lib);
  "consciousness engine"/HoTT are documentation intent, unexercised by committed code.
- Risk: readers trust marketing over the (partially correct) math.
- Mitigate: ESHKOL_STRUCTURE.md separates fact (code) from intent (docs).

**Claim C:** "Poincaré formulas agree with GRR/qLLM dylib geometry."
- Attack: header admits "up to convention; dylib is source of truth" — i.e. the
  `.esk` is a mirror with known drift. Our cross-check found the exp-map drift is a
  *bug*, not just convention.
- Mitigate: cross-validation keeps both forms (eshkol-verbatim + corrected).

## PHASE 3 — Triple Synthesis

eshkol's geometry is **mostly right and one part wrong**. The curvature tensor
(math) is sound; the exp-map (a specific use) carries a conformal-factor bug.
WuBuMath's existing `exp_0` is the consistent reference. The fix is small and
localized. This is an *invitation*: a clean upstream PR (with our reproducible
test) turns a finding into collaboration, not confrontation.

## PHASE 4 — Risk of OUR audit

- Confirmation bias: we compared against WuBuMath (our own lib). Risk we inherited
  WuBuMath's convention. **Mitigated:** we reproduced the bug two independent ways
  (eshkol's own distance + invariant), and WuBuMath's `exp_0` is provably
  self-consistent (`dist(0,exp_0(v))=2|v|`).
- Measurement error: finite-diff Christoffel check has tolerance 5e-2. **Mitigated:**
  the exp-map bug is a separate, exact (algebraic) failure of the invariant.

## WuBuMath Proof Extension (the bridge)

Our `lean/WubuProofs/` already proves the surrounding theory — eshkol's work can
*cite* these:
- `PoincareBall.lean`: `dist_from_origin_formula`, `poincare_metric_is_conformal`,
  `curvature_scaling` (0 sorry).
- `FiberBundle.lean`: `so3_contains_identity`, `so3_closed_under_compose`,
  `wubu_is_principal_bundle` (0 sorry) — relevant if eshkol adds rotation manifolds.
- `NestedHyperbolicSpaces.lean`: `nested_balls`, `phi_curvature_progression` (0 sorry).

Proposal: port eshkel's corrected exp-map into WuBuMath as `exp_p` (general base),
then add a Lean lemma `exp_map_geodesic_invariant` proving `dist(p,exp_p(v))=|v|`
(in WuBuMath's `2|v|` convention) — closing the loop from numeric to formal.

## Invitation to tsotchke (rigorous but welcoming)
"We cross-validated your manifold.esk against an independent C implementation.
Your Christoffel/curvature are correct. We found one bug in exp-map (conformal
factor inside tanh) — reproduced here: `cross-validation/`. Fix + test attached.
Want to co-author a PR? We can also formalize the invariant in Lean as a shared
proof artifact."
