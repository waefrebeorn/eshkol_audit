# eshkol_audit — Overnight Map (Jul 11 2026)

## Quick Trunk
- Fork: /home/wubu/eshkol-fork → waefrebeorn/eshkol_audit (master)
- Upstream: tsotchke/eshkol (d861d20a)
- Audit C: cross-validation/{wubu_poincare_geom.{c,h}, test_crossval.c, REPORT.md}
- Reference: waefrebeorn/WuBuMath (lean/WubuProofs/, src/math/wubu_poincare_geom.c)

## Where We Are
- VERIFIED: eshkol exp-map conformal-factor bug reproduced; Christoffel/curvature correct.
- NOT VERIFIED: REPL build (LLVM 21 absent); "consciousness engine"/HoTT (doc only).

## Workstreams
A — upstream PR fixing manifold-exp-map (highest impact, invites collaboration)
B — extend cross-validation to sphere + product manifolds
C — bridge to FiberBundle.lean SO(3) proof

## Data Not To Re-derive
- eshkol exp-map bug is in `cross-validation/REPORT.md` §F1 (reproducible command).
- Build needs LLVM 21; audit math needs only gcc.

## Fallback
If upstream unresponsive: keep fork as evidence trail; document in AUDIT-style md.
