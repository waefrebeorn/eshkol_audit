static float si_randf(void) { /* uniform in [-g_si_scale, g_si_scale] */
    g_si_rng = g_si_rng*6364136223846793005UL + 1442695040888963407UL;
    unsigned int v = (unsigned int)(g_si_rng >> 33);
    return ((float)v / (float)0x7fffffffu - 1.0f) * g_si_scale;
}

/* Gradient-check context shared with gcheck_kind. */
typedef struct {
    InterpreterWeights* w;
    float* state;
    float (*pe)[D];
    int np;
    const float*  target;    /* float target (for the float32-forward display)  */
    const double* targetd;   /* same values in double (for the double-FD verdict)*/
    float* next;             /* scratch for the float32 forward output          */
    float eps;
    float max_rel;
    int   all_pass;
} GCheckCtx;

/** @brief float32 squared-error loss via the artifact's actual forward pass
 *         (forward_with_weights()), used for display only. */
static double gcheck_loss_f(GCheckCtx* gc) {
    forward_with_weights(gc->w, gc->state, gc->pe, gc->np, gc->next);
    double L = 0.0;
    for (int i = 0; i < D; i++) { double d = (double)gc->next[i] - gc->target[i]; L += 0.5*d*d; }
    return L;
}
/** @brief Double-precision reference loss (forward_loss_double()); this is
 *         the value that actually drives the gradient check's PASS/FAIL
 *         verdict. */
static double gcheck_loss_d(GCheckCtx* gc) {
    return forward_loss_double(gc->w, gc->state, gc->pe, gc->np, gc->targetd);
}

/**
 * @brief Finite-difference gradient check for one weight array kind. Picks
 *        the @p nsample indices (up to 8) in @p warr/@p garr (of length
 *        @p n) with the largest analytic gradient magnitude — where central
 *        finite differences in float32 carry real signal — and compares
 *        the analytic gradient against both a float32 FD (display only)
 *        and the double-precision reference FD (drives PASS/FAIL, via
 *        gcheck_loss_f()/gcheck_loss_d()). Updates gc->max_rel and
 *        gc->all_pass, and prints one diagnostic line per sampled index.
 */
static void gcheck_kind(GCheckCtx* gc, const char* kind,
                        float* warr, const float* garr, int n, int nsample) {
    /* find top-|grad| indices via simple selection */
    int idx[8]; if (nsample > 8) nsample = 8;
    for (int s = 0; s < nsample; s++) {
        int best = -1; float bestv = -1.0f;
        for (int i = 0; i < n; i++) {
            int dup = 0; for (int t = 0; t < s; t++) if (idx[t] == i) { dup = 1; break; }
            if (dup) continue;
            float a = fabsf(garr[i]);
            if (a > bestv) { bestv = a; best = i; }
        }
        idx[s] = best;
    }
    double eps = (double)gc->eps;
    for (int s = 0; s < nsample; s++) {
        int i = idx[s];
        float* wp = &warr[i];
        float ga = garr[i];
        float orig = *wp;
        /* float32 FD (displayed) via the artifact's actual float forward */
        *wp = orig + gc->eps; double Lpf = gcheck_loss_f(gc);
        *wp = orig - gc->eps; double Lmf = gcheck_loss_f(gc);
        /* double-precision reference FD (drives the verdict) */
        *wp = orig + gc->eps; double Lpd = gcheck_loss_d(gc);
        *wp = orig - gc->eps; double Lmd = gcheck_loss_d(gc);
        *wp = orig;
        float fd_f = (float)((Lpf - Lmf) / (2.0 * eps));
        float fd_d = (float)((Lpd - Lmd) / (2.0 * eps));
        float denom = fabsf(ga) + fabsf(fd_d) + 1e-7f;
        float rel = fabsf(ga - fd_d) / denom;   /* analytic vs DOUBLE FD */
        if (rel > gc->max_rel) gc->max_rel = rel;
        int fail = (rel >= 1e-2f);
        if (fail) gc->all_pass = 0;
        printf("  %-14s[%6d] %13.6f %13.6f %13.6f %11.2e%s\n",
               kind, i, ga, fd_d, fd_f, rel, fail ? "  <-- FAIL" : "");
    }
}

/**
 * @brief Self-improvement demo: initializes fresh randomized interpreter
 *        weights (scaled per-matrix so the stacked SQUARE FFN layers stay
 *        numerically bounded), verifies the analytic backprop
 *        (backward_through_weights()) against a finite-difference gradient
 *        check (gcheck_kind()) on a sample program, then runs gradient
 *        descent (apply_weight_gradient_step()) to demonstrate the program
 *        weights can be trained to reduce loss on the target output.
 */
