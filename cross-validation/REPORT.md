# Cross-Validation Report: eshkol/manifold.esk vs WuBuMath

**Audit date:** 2026-07-11
**Auditor:** waefrebeorn (triple devil's-advocate cross-check)
**Subjects:**
- `lib/core/manifold.esk` (fork HEAD `d55adf8e`; exp-map at `57bbfc85`, "real Euclidean/Poincaré/spherical geometry")
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

### F1 — eshkol `manifold-exp-map` violates the geodesic invariant (BUG — root cause CORRECTED)

eshkol's released exp-map (`lib/core/manifold.esk:107-109`):
```
lam    = 2 / (1 - |p|^2)
factor = tanh(0.5 * lam * |v|) / |v|
exp_p(v) = p ⊕ (factor · v)
```
**Triple-DA re-check (2026-07-11, revision):** my earlier root-cause claim —
"the conformal factor `lam` is placed *inside* the tanh, that's the bug" — is
**WRONG / misleading**. The `tanh(λ|v|/2)` form is the textbook-correct
Nickel–Kiela Poincaré exp map; `lam` belongs inside the tanh. The real defect
is a **tangent-norm scaling mismatch between the exp-map and the distance
formula**, exposed only OFF the origin:

- At `p = 0`, eshkol's exp-map + distance give `dist(0, exp_0(v)) = 2|v|`
  *exactly* (verified, `da_eshkol_exp.c` origin rows). So at the origin the
  tangent norm is implicitly a **chordal/half-distance** quantity, not the
  standard hyperbolic Riemannian norm (`2·atanh(|v|)`).
- For `p ≠ 0`, the same convention is NOT preserved: the conformal prefactor
  `lam(p)` is folded into the *magnitude* of the tangent vector, but the
  distance formula measures the ambient chord without the matching rescaling.
  Result: `dist(p, exp_p(v)) / |v|` drifts from **2.83 → 4.96** as `|v|`
  grows (verified independently in `da_eshkol_exp.c`, faithful port of
  `manifold.esk:107-109` + the arccosh distance, no dependency on the
  cross-validation port). Drift > 0.15 ⇒ the map is not a consistent geodesic
  exponential off the origin.

**Reproduced:** `cross-validation/test_crossval.c` check #1 (`[FAIL-expected]`)
and independently `da_eshkol_exp.c`. The non-constancy is real; the prior
"lam-in-tanh" explanation was the wrong mechanism.

**Reference fix (not applied to eshkol's file — kept verbatim for evidence):**
make the tangent norm Riemannian-consistent with the distance formula, i.e.
exp-map and distance must agree on what `|v|` means at every base `p`
(either rescale the ambient `dist` by `2·atanh`, or normalize the exp-map's
tangent by `lam(p)` so the chord equals the metric distance). The fix is a
*convention reconciliation*, not a tanh-argument edit.

### F2 — WuBuMath's own exp_0 + distance are self-consistent (caveat)

WuBuMath's existing `wubu_expmap` (single `tanh(√c·|v|)/|v|`, no per-base `lam`)
combined with `wubu_hyperbolic_distance` gives `dist(0, exp_0(v)) = 2|v|`
**exactly** — the same chordal/half-distance convention eshkol uses at the
origin. So WuBuMath is *internally* consistent at the origin, and (because it
has no per-base conformal prefactor at all) it does **not** suffer the
off-origin drift that eshkol's `lam(p)`-scaled map does.

Caveat (3×DA honesty): "WuBuMath correct, eshkol wrong" overstates it. Both
share the `dist=2|v|`-at-origin convention. The defect is specifically
eshkol's **off-origin extension** of that convention via `lam(p)` without a
matching rescale in the distance formula. WuBuMath sidesteps it only because
its exp-map never leaves the origin (`expmap_0`). A WuBuMath `exp_p` for
arbitrary base `p` would need the same convention reconciliation.

### F3 — eshkol's analytic Christoffel symbols (PASS 2 caveat)

`manifold-christoffel` (and the derived sectional/Ricci/scalar/Riemann
curvature) are the standard conformal-formula symbols. The cross-validation's
check #3 (Christoffel vs finite-difference geodesic) was reported as a
**"convention-doc, not a fail"** line with `maxgap = 0.056` — i.e. a *hand
gap of 0.056 between analytic Γ and the FD of the (buggy) exp map*. That gap
is expected given F1 (the exp-map is not the true geodesic), so it is NOT
evidence the Christoffel symbols themselves are wrong. To actually validate
them, one must FD the *true* geodesic (e.g. RK4 on `x''=−Γx'x'`), not FD the
buggy exp-map. WuBuMath's independent RK4 geodesic (`wubu_manifold.c`) agreed
with its own analytic Γ to 5e-2, which is the real supporting evidence — but
that validates WuBuMath's Γ, not eshkol's. **Open:** port WuBuMath's RK4 to
eshkol's metric and FD eshkol's Γ to close this. Currently F3 is
"Christoffel formulas are the standard ones; rigorous numerical cross-check
vs eshkol's own symbols not yet done."

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
- `da_eshkol_exp.c` — independent re-derivation of the F1 non-constancy (faithful
  port of `manifold.esk:107-109` + the arccosh distance, no dependency on the
  port above). Build: `gcc -std=c11 -O3 da_eshkol_exp.c -o da_eshkol_exp -lm && ./da_eshkol_exp`
- `REPORT.md` — this document.

## Verdict

eshkol's geometry library has **correct curvature tensor / Christoffel /
distance formulas**, but its `manifold-exp-map` is **not a consistent geodesic
exponential off the origin**: the tangent norm is defined chordally
(`dist(0,exp_0(v))=2|v|` exactly) at the origin, and the per-base conformal
prefactor `lam(p)` is folded into the tangent magnitude without a matching
rescale in the distance formula, so `dist(p,exp_p(v))/|v|` drifts 2.83→4.96
for `p≠0` (F1, reproduced two ways). This is a **convention-mismatch bug**, not
a "wrong tanh argument." Christoffel/curvature (F3/F4) are the standard
formulas but were only *symbolically* checked, not numerically cross-validated
against eshkol's own geodesic — that cross-check is open. WuBuMath's exp/distance
share the origin convention and avoid the drift only because they never leave
the origin. Recommend an upstream PR reconciling exp-map vs distance norm; the
fork keeps the buggy form verbatim as evidence.
