# eshkol_audit — Goal Mantra (paste at session start)

Path: /home/wubu/eshkol-fork | Remote: waefrebeorn/eshkol_audit (upstream=tsotchke/eshkol)
Build audit (no LLVM): cd cross-validation && gcc -std=c11 -O3 -I. wubu_poincare_geom.c test_crossval.c -o crossval -lm && ./crossval

=== STATE ===
✅ Verified: eshkol manifold-exp-map bug reproduced (geodesic invariant fails p≠0)
✅ Verified: analytic Christoffel/curvature (K/R/Ricci) correct
✅ Verified: WuBuMath exp_0 + distance self-consistent (2|v|)
⚠️ Broken:   can't build REPL (no LLVM 21 here) — audit via standalone C only
⚠️ Debatable: "consciousness engine"/HoTT claims unexercised (doc intent only)

=== STREAMS ===
S1 [P0] upstream PR: fix manifold-exp-map (conformal factor scales vector, not in tanh)
S2 [P1] extend cross-validation to sphere + Euclidean×Poincaré product manifolds
S3 [P1] cross-check eshkol rotation ops vs our FiberBundle.lean SO(3) proof
S4 [P2] document unproven "consciousness/HoTT" claims as needs-LLVM-21

=== THE LOOP ===
pick highest undone → reproduce in cross-validation/ → verify numeric invariant →
document in REPORT.md → keep upstream verbatim → report. NO silent fixes.

=== FULL CONTEXT ===
Read .hermes/mind-palace/prestige_prompt.md
