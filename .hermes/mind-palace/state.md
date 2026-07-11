# eshkol_audit — State (Jul 11 2026 v1)

## Binary / Artifact Truth Table
| Artifact | Claimed | Actual | Evidence |
|---|---|---|---|
| eshkol REPL | builds | ❌ cannot (no LLVM 21 in env) | cmake needs LLVM 21 |
| cross-validation/crossval | runs | ✅ builds+passes | gcc; ./crossval exit 0 |
| manifold.esk exp-map | geodesic map | ⚠️ bug for p≠0 | crossval check #1 |
| manifold.esk Christoffel | analytic | ✅ correct | matches RK4 accel |
| manifold.esk curvature | K/R/Ricci | ✅ correct | crossval checks #4,#5 |

## Fixed / Found
- **BUG (reproduced):** `manifold-exp-map` puts `lam=2/(1−|p|²)` inside `tanh` →
  violates `dist(p,exp_p(v))=const·|v|` for p≠0. Fix: scale vector, not tanh arg.
- **CORRECT:** analytic Christoffel symbols; curvature tensor; distance formula.

## Hidden State
- `vm_geometric.c` referenced in docs does NOT exist in git tree.
- eshkol exp/distance use different conventions (dylib is "source of truth" per header).

## Verified Components
- Poincaré exp_0 (origin), geodesic distance, Christoffel, sectional/Ricci/scalar curvature.
