# Cross-Validation Report: eshkol/manifold.esk vs WuBuMath

**Audit date:** 2026-07-11
**Auditor:** waefrebeorn (triple devil's-advocate cross-check)
**Subjects:**
- `tsotchke/eshkol` `lib/core/manifold.esk` (release `d861d20a`, "v1.3.3-evolve")
- `waefrebeorn/WuBuMath` `src/math/wubu_poincare_geom.c` (commit `ac21a24`)

## Method

Two independent implementations of constant-curvature Riemannian geometry
(Euclidean / Poincaré K=−1 / sphere K=+1) are compared against:
1. Each other (ported formulas).
2. The manifold's own geometric invariants (geodesic distance, Christoffel =
   geodesic 2nd derivative, constant curvature).

No theorem without a check. No "proof" without a numerical or symbolic
verification.

## Findings

### F1 — eshkol `manifold-exp-map` violates the geodesic invariant (BUG)

eshkol's released exp-map:
```
factor = tanh(0.5 * lam * |v|) / |v|,   lam = 2/(1 - |p|^2)
exp_p(v) = p ⊕ (factor · v)
```
The conformal factor `lam` is placed **inside** the `tanh`. Cross-validation
against eshkol's own distance formula
`dist(a,b) = arccosh(1 + 2|a⊕(−b)|²/((1−|a|²)(1−|b|²)))` shows the geodesic
invariant `dist(p, exp_p(v)) = const·|v|` does **NOT** hold for `p ≠ 0`:
the ratio `dist/|v|` drifts >0.15 across random bases (measured 2.1 → 3.1 as
`|v|` grows). The map overshoots.

**Reproduced:** `cross-validation/test_crossval.c` check #1 (the
`[FAIL-expected]` line is the intentional reproduction of the bug).

**Reference fix (not applied to eshkol's file — kept verbatim for evidence):**
the conformal factor must scale the *vector*, not sit inside the tanh:
```
exp_p(v) = p ⊕ ( lam · tanh(|v|/2)/|v| · v )
```

### F2 — WuBuMath's own exp_0 + distance are self-consistent

WuBuMath's existing `wubu_expmap` (single `tanh(|v|)/|v|`, no `lam`) combined
with `wubu_hyperbolic_distance` satisfies `dist(0, exp_0(v)) = 2|v|` exactly.
WuBuMath was correct; eshkol's released exp_p was not.

### F3 — eshkol's analytic Christoffel symbols are internally correct

`manifold-christoffel` (and the derived sectional/Ricci/scalar/Riemann
curvature) are the standard conformal-formula symbols and **agree with
WuBuMath's independent RK4 geodesic acceleration** `x''(0) = −Γ(p)[v,v]` to
5e-2 in WuBuMath's consistent convention. The *curvature tensor math* is sound;
only the *exp-map* (a specific use of it) carries the bug.

### F4 — Analytic curvature verified

K = −1 (Poincaré), +1 (sphere), 0 (Euclidean); scalar R = K·n(n−1);
Ricci_ij = K(n−1)·g_ij all confirmed.

## How to run (no LLVM needed)

```bash
cd cross-validation
gcc -std=c11 -O3 -I. wubu_poincare_geom.c test_crossval.c -o crossval -lm
./crossval
```

## Files

- `wubu_poincare_geom.h` / `.c` — standalone C port of manifold.esk
  (eshkol-verbatim `poincare_exp_eshkol` + corrected `poincare_exp_correct`).
- `test_crossval.c` — the checks above.
- `REPORT.md` — this document.

## Verdict

eshkol's geometry library is **mostly correct** (curvature tensor, distance,
Christoffel) but ships a **conformal-factor bug in `manifold-exp-map`** that
breaks the geodesic invariant off the origin. WuBuMath's existing exp/distance
were the consistent reference. Recommend upstream PR fixing the exp-map; the
fork keeps the buggy form verbatim as evidence.
