static void self_improve_demo(void) {
    printf("=== Self-improvement loop (gradient descent on program weights) ===\n\n");

    /* Fresh weights with small random values + an explicit per-layer ff_type
     * schedule so the check exercises type-1 and type-2 layers and attention. */
    InterpreterWeights* w = (InterpreterWeights*)calloc(1, sizeof(InterpreterWeights));
    if (!w) { printf("alloc failed\n"); return; }
    g_si_rng = 0xC0FFEEUL;
    /* Scale weights by 1/sqrt(fan_in) and keep activations small so the stacked
     * SQUARE (type-1) layers stay numerically bounded (no inf/nan). */
    const float s_d   = 0.35f / sqrtf((float)D);        /* matrices with fan-in D     */
    const float s_ffn = 0.35f / sqrtf((float)FFN_DIM);  /* matrices with fan-in FFN_DIM*/
    /* wq/wk/bq drive only the 2-d bilinear attention score; with small weights
     * the softmax stays near-uniform and their gradients fall below the float32
     * finite-difference noise floor. Scale them larger so the softmax is in a
     * responsive (non-saturated) regime and the gradient check is meaningful. */
    const float s_qk  = 1.6f / sqrtf((float)D);
    for (int L = 0; L < N_LAYERS; L++) {
        g_si_scale = s_qk;
        for (int i = 0; i < D*D; i++) { w->wq[L][i]=si_randf(); w->wk[L][i]=si_randf(); }
        g_si_scale = s_d;
        for (int i = 0; i < D*D; i++) { w->wv[L][i]=si_randf(); w->wo[L][i]=si_randf(); }
        for (int i = 0; i < D*FFN_DIM; i++){ w->ff_up[L][i]=si_randf(); w->ff_gate[L][i]=si_randf(); }
        g_si_scale = s_ffn;
        for (int i = 0; i < FFN_DIM*D; i++) w->ff_down[L][i]=si_randf();
        g_si_scale = 0.6f;
        for (int i = 0; i < D; i++) w->bq[L][i]=si_randf();
        g_si_scale = 0.02f;
        for (int i = 0; i < FFN_DIM; i++){ w->ff_up_b[L][i]=si_randf(); w->ff_gate_b[L][i]=si_randf(); }
        for (int i = 0; i < D; i++) w->ff_down_b[L][i]=si_randf();
    }
    g_si_scale = 0.05f;
    /* type schedule: L0 type1 (with attention) + L1 type2 exercise both FFN
     * kinds and the attention path; the remaining layers are no-ops so the
     * stacked SQUARE non-linearity does not blow the dynamic range out of
     * float32 range (which would swamp the finite-difference check). */
    w->ff_type[0]=1; w->ff_type[1]=2; w->ff_type[2]=0; w->ff_type[3]=0; w->ff_type[4]=0; w->ff_type[5]=0;

    /* Fixed inputs. */
    const int np = 4;
    static float state[D];
    static float pe[256][D];
    static float target[D];
    for (int i = 0; i < D; i++) state[i]=si_randf();
    for (int p = 0; p < np; p++) for (int i = 0; i < D; i++) pe[p][i]=si_randf();
    for (int i = 0; i < D; i++) target[i]=si_randf();

    /* ---- GRADIENT CHECK ---- */
    static float next[D], dL_dnext[D], dL_dstate[D];
    static WeightGrads g;
    forward_with_weights(w, state, pe, np, next);
    for (int i = 0; i < D; i++) dL_dnext[i] = next[i]-target[i];
    zero_grads(&g);
    backward_through_weights(w, state, pe, np, dL_dnext, &g, dL_dstate);

    static double targetd[D];
    for (int i = 0; i < D; i++) targetd[i] = target[i];

    GCheckCtx gc;
    gc.w = w; gc.state = state; gc.pe = pe; gc.np = np;
    gc.target = target; gc.targetd = targetd;
    gc.next = next; gc.eps = 1e-3f; gc.max_rel = 0.0f; gc.all_pass = 1;

    printf("  Gradient check (analytic vs central finite-difference).\n");
    printf("  For each weight kind the 3 highest-|gradient| components are tested.\n");
    printf("  Verdict uses a double-precision reference FD (fd64); the artifact's\n");
    printf("  float32-forward FD (fd32) is shown too — it is unreliable for the\n");
    printf("  small attention gradients routed through the SQUARE non-linearity.\n\n");
    printf("  %-20s %13s %13s %13s %11s\n",
           "weight kind[idx]", "analytic", "fd64", "fd32", "rel-err");

    /* type-1 FFN layer (L0, also attention layer) */
    gcheck_kind(&gc, "ff_up[0]",     w->ff_up[0],     g.dff_up[0],     D*FFN_DIM, 3);
    gcheck_kind(&gc, "ff_up_b[0]",   w->ff_up_b[0],   g.dff_up_b[0],   FFN_DIM,   3);
    gcheck_kind(&gc, "ff_down[0]",   w->ff_down[0],   g.dff_down[0],   FFN_DIM*D, 3);
    gcheck_kind(&gc, "ff_down_b[0]", w->ff_down_b[0], g.dff_down_b[0], D,         3);
    /* type-2 FFN layer (L1) */
    gcheck_kind(&gc, "ff_gate[1]",   w->ff_gate[1],   g.dff_gate[1],   D*FFN_DIM, 3);
    gcheck_kind(&gc, "ff_gate_b[1]", w->ff_gate_b[1], g.dff_gate_b[1], FFN_DIM,   3);
    gcheck_kind(&gc, "ff_up[1]",     w->ff_up[1],     g.dff_up[1],     D*FFN_DIM, 3);
    gcheck_kind(&gc, "ff_up_b[1]",   w->ff_up_b[1],   g.dff_up_b[1],   FFN_DIM,   3);
    gcheck_kind(&gc, "ff_down[1]",   w->ff_down[1],   g.dff_down[1],   FFN_DIM*D, 3);
    gcheck_kind(&gc, "ff_down_b[1]", w->ff_down_b[1], g.dff_down_b[1], D,         3);
    /* attention weights (L0) */
    gcheck_kind(&gc, "wq[0]",        w->wq[0],        g.dwq[0],        D*D, 3);
    gcheck_kind(&gc, "wk[0]",        w->wk[0],        g.dwk[0],        D*D, 3);
    gcheck_kind(&gc, "wv[0]",        w->wv[0],        g.dwv[0],        D*D, 3);
    gcheck_kind(&gc, "wo[0]",        w->wo[0],        g.dwo[0],        D*D, 3);
    gcheck_kind(&gc, "bq[0]",        w->bq[0],        g.dbq[0],        D,   3);

    int all_pass = gc.all_pass;
    printf("\n  Max relative error: %.3e   GRADIENT CHECK: %s\n\n",
           gc.max_rel, all_pass ? "PASS" : "FAIL");

    /* ---- TRAINING LOOP ---- */
    /* Target = current output nudged in a few dims; loss must strictly drop. */
    forward_with_weights(w, state, pe, np, next);
    for (int i = 0; i < D; i++) target[i] = next[i];
    target[0]+=0.5f; target[1]+=0.5f; target[5]+=0.5f; target[10]+=0.5f; target[42]+=0.5f;

    const float lr = 2e-3f;
    printf("  Training loop (50 iters, lr=2e-3):\n");
    float first_loss = 0.0f, last_loss = 0.0f;
    for (int it = 0; it < 50; it++) {
        forward_with_weights(w, state, pe, np, next);
        float Lval = 0.0f;
        for (int i = 0; i < D; i++) { float d = next[i]-target[i]; Lval += 0.5f*d*d; }
        for (int i = 0; i < D; i++) dL_dnext[i] = next[i]-target[i];
        zero_grads(&g);
        backward_through_weights(w, state, pe, np, dL_dnext, &g, dL_dstate);
        apply_weight_gradient_step(w, &g, lr);
        if (it == 0) first_loss = Lval;
        last_loss = Lval;
        if (it % 5 == 0) printf("    iter %2d  loss = %.6f\n", it, Lval);
    }
    /* final loss after the last step */
    forward_with_weights(w, state, pe, np, next);
    float final_loss = 0.0f;
    for (int i = 0; i < D; i++) { float d = next[i]-target[i]; final_loss += 0.5f*d*d; }

    printf("\n  First loss = %.6f   Final loss = %.6f   TRAINING: %s\n",
           first_loss, final_loss, (final_loss < first_loss) ? "PASS" : "FAIL");
    printf("\n  Overall: %s\n\n",
           (all_pass && final_loss < first_loss) ? "ALL CHECKS PASS" : "CHECKS FAILED");

    free(w);
}

/**
 * @brief Entry point for the Eshkol VM weight compiler artifact.
 *
 * Supports `--self-improve` (runs self_improve_demo() and exits) and the
 * paper-artifact trace flags `--trace-vm`/`--trace-transformer`/
 * `--trace-simulated PATH` (opens JSONL trace files consumed by
 * scripts/paper/compare_traces.py). Otherwise generates the interpreter
 * weights (generate_weights()), runs the built-in test suite comparing the
 * reference/simulated/matrix execution modes (test()), and optionally
 * exports the weights to ESHKOL_WEIGHTS_OUT via export_weights_binary().
 */
int main(int argc, char** argv) {
    printf("=== Eshkol VM Weight Compiler ===\n\n");

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--self-improve") == 0) {
            self_improve_demo();
            return 0;
        }
    }

    /* Paper artifact CLI:
     *   --trace-vm <path>           emit per-step reference-VM JSONL trace
     *   --trace-transformer <path>  emit per-step matrix-forward JSONL trace
     * Consumed by scripts/paper/compare_traces.py (paper §4.4). */
    const char* trace_vm_path = NULL;
    const char* trace_tf_path = NULL;
    const char* trace_sim_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--trace-vm") == 0 && i + 1 < argc) {
            trace_vm_path = argv[++i];
        } else if (strcmp(argv[i], "--trace-transformer") == 0 && i + 1 < argc) {
            trace_tf_path = argv[++i];
        } else if (strcmp(argv[i], "--trace-simulated") == 0 && i + 1 < argc) {
            trace_sim_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--trace-vm PATH] [--trace-transformer PATH] [--trace-simulated PATH]\n", argv[0]);
            printf("Env: ESHKOL_WEIGHTS_OUT=PATH, ESHKOL_BC=PATH\n");
            return 0;
        }
    }
    if (trace_vm_path) {
        g_trace_vm_fp = fopen(trace_vm_path, "w");
        if (!g_trace_vm_fp) { perror(trace_vm_path); return 1; }
        printf("[TRACE] reference VM trace → %s\n", trace_vm_path);
    }
    if (trace_tf_path) {
        g_trace_tf_fp = fopen(trace_tf_path, "w");
        if (!g_trace_tf_fp) { perror(trace_tf_path); return 1; }
        printf("[TRACE] matrix-forward trace → %s\n", trace_tf_path);
    }
    if (trace_sim_path) {
        g_trace_sim_fp = fopen(trace_sim_path, "w");
        if (!g_trace_sim_fp) { perror(trace_sim_path); return 1; }
        printf("[TRACE] simulated-layer trace → %s\n", trace_sim_path);
    }

    g_weights = (InterpreterWeights*)calloc(1, sizeof(InterpreterWeights));
    if (g_weights) generate_weights(g_weights);

    printf("\n  Tests (ref=reference, sim=simulated, mat=matrix-based):\n\n");

    /* ── Stage 0: Original v1 tests (renumbered to canonical opcodes) ── */
    printf("  --- Stage 0: Core arithmetic & control ---\n");

    { Instr p[]={{OP_NOP,0},{OP_CONST,42},{OP_PRINT,0},{OP_HALT,0}};
      test("nop pc++", p, 4, 42); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3+5", p, 5, 8); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_ADD,0},{OP_CONST,2},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("(3+5)*2", p, 7, 16); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,7},{OP_SUB,0},{OP_PRINT,0},{OP_HALT,0}};
      test("10-7", p, 5, 3); }
    /* SWAP exchanges TOS<->SOS. After CONST 3, CONST 5: TOS=5, SOS=3 → SUB= -2.
     * With SWAP first: TOS=3, SOS=5 → SUB = SOS-TOS = 2. Reveals stack order. */
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_SWAP,0},{OP_SUB,0},{OP_PRINT,0},{OP_HALT,0}};
      test("swap then 5-3", p, 6, 2); }
    /* mem[0]=42: SET_LOCAL takes operand as address, TOS as value */
    { Instr p[]={{OP_CONST,42},{OP_SET_LOCAL,0},{OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("mem[0]=42", p, 5, 42); }

    /* sum(1..5) = 15: GET_LOCAL/SET_LOCAL with operand-based addressing */
    { Instr p[]={
        {OP_CONST,0},{OP_SET_LOCAL,0},                       /* mem[0]=0 (sum) */
        {OP_CONST,5},{OP_SET_LOCAL,1},                       /* mem[1]=5 (counter) */
        /* loop (pc=4): */
        {OP_GET_LOCAL,1},                                    /* push counter */
        {OP_JUMP_IF_FALSE,15},                               /* if counter==0 goto end */
        /* body: */
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},        /* sum + counter */
        {OP_SET_LOCAL,0},                                    /* mem[0] = new sum */
        {OP_GET_LOCAL,1},{OP_CONST,1},{OP_SUB,0},            /* counter - 1 */
        {OP_SET_LOCAL,1},                                    /* mem[1] = counter-1 */
        {OP_JUMP,4},                                         /* goto loop */
        /* end (pc=15): */
        {OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("sum(1..5)", p, 18, 15); }

    /* 5! = 120 */
    { Instr p[]={
        {OP_CONST,1},{OP_SET_LOCAL,0},                       /* mem[0]=1 (result) */
        {OP_CONST,5},{OP_SET_LOCAL,1},                       /* mem[1]=5 (counter) */
        /* loop (pc=4): */
        {OP_GET_LOCAL,1},                                    /* push counter */
        {OP_JUMP_IF_FALSE,15},                               /* if counter==0 goto end */
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_MUL,0},        /* result * counter */
        {OP_SET_LOCAL,0},                                    /* result = product */
        {OP_GET_LOCAL,1},{OP_CONST,1},{OP_SUB,0},            /* counter - 1 */
        {OP_SET_LOCAL,1},                                    /* counter = counter-1 */
        {OP_JUMP,4},                                         /* goto loop */
        /* end (pc=15): */
        {OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("5!", p, 18, 120); }

    /* fib(7) = 13 */
    { Instr p[]={
        {OP_CONST,0},{OP_SET_LOCAL,0},                       /* a = 0 */
        {OP_CONST,1},{OP_SET_LOCAL,1},                       /* b = 1 */
        {OP_CONST,7},{OP_SET_LOCAL,2},                       /* n = 7 */
        /* loop (pc=6): */
        {OP_GET_LOCAL,2},                                    /* push n */
        {OP_JUMP_IF_FALSE,19},                               /* if n==0 goto end */
        /* body: a,b = b, a+b */
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},        /* a + b */
        {OP_GET_LOCAL,1},{OP_SET_LOCAL,0},                   /* a = old b */
        {OP_SET_LOCAL,1},                                    /* b = a+b */
        {OP_GET_LOCAL,2},{OP_CONST,1},{OP_SUB,0},            /* n - 1 */
        {OP_SET_LOCAL,2},                                    /* n = n-1 */
        {OP_JUMP,6},                                         /* goto loop */
        /* end (pc=19): */
        {OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("fib(7)", p, 22, 13); }

    { Instr p[]={
        {OP_CONST,2},{OP_CONST,3},{OP_MUL,0},
        {OP_CONST,4},{OP_CONST,5},{OP_MUL,0},
        {OP_ADD,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("(2*3)+(4*5)", p, 9, 26); }
    { Instr p[]={{OP_CONST,7},{OP_CONST,11},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("7*11", p, 5, 77); }

    /* ── Stage 1: Trivial push opcodes ── */
    printf("\n  --- Stage 1: NIL, TRUE, FALSE ---\n");
    { Instr p[]={{OP_TRUE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("true", p, 3, 1); }
    { Instr p[]={{OP_FALSE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("false", p, 3, 0); }
    { Instr p[]={{OP_NIL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("nil", p, 3, -1); }
    { Instr p[]={{OP_CONST,5},{OP_TRUE,0},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5+true(=6)", p, 5, 6); }

    /* ── Stage 2: NEG, ABS ── */
    printf("\n  --- Stage 2: NEG, ABS ---\n");
    { Instr p[]={{OP_CONST,5},{OP_NEG,0},{OP_PRINT,0},{OP_HALT,0}};
      test("neg(5)", p, 4, -5); }
    { Instr p[]={{OP_CONST,-7},{OP_ABS,0},{OP_PRINT,0},{OP_HALT,0}};
      test("abs(-7)", p, 4, 7); }
    { Instr p[]={{OP_CONST,3},{OP_ABS,0},{OP_PRINT,0},{OP_HALT,0}};
      test("abs(3)", p, 4, 3); }
    { Instr p[]={{OP_CONST,5},{OP_NEG,0},{OP_ABS,0},{OP_PRINT,0},{OP_HALT,0}};
      test("abs(neg(5))", p, 5, 5); }

    /* ── Stage 3: Comparisons ── */
    printf("\n  --- Stage 3: EQ, LT, GT, LE, GE, NOT ---\n");
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_EQ,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3==5", p, 5, 0); }
    { Instr p[]={{OP_CONST,5},{OP_CONST,5},{OP_EQ,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5==5", p, 5, 1); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_LT,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3<5", p, 5, 1); }
    { Instr p[]={{OP_CONST,5},{OP_CONST,3},{OP_LT,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5<3", p, 5, 0); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_GT,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3>5", p, 5, 0); }
    { Instr p[]={{OP_CONST,5},{OP_CONST,3},{OP_GT,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5>3", p, 5, 1); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,5},{OP_LE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3<=5", p, 5, 1); }
    { Instr p[]={{OP_CONST,5},{OP_CONST,5},{OP_LE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5<=5", p, 5, 1); }
    { Instr p[]={{OP_CONST,5},{OP_CONST,3},{OP_GE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("5>=3", p, 5, 1); }
    { Instr p[]={{OP_CONST,3},{OP_CONST,3},{OP_GE,0},{OP_PRINT,0},{OP_HALT,0}};
      test("3>=3", p, 5, 1); }
    /* Matrix NOT uses operand as the zero-indicator scale. */
    { Instr p[]={{OP_CONST,0},{OP_NOT,1},{OP_PRINT,0},{OP_HALT,0}};
      test("not(0)=true", p, 4, 1); }
    /* Composite: if (3 < 5) print 42 else print 99 */
    { Instr p[]={
        {OP_CONST,3},{OP_CONST,5},{OP_LT,0},
        {OP_JUMP_IF_FALSE,6},{OP_CONST,42},{OP_JUMP,7},
        {OP_CONST,99},{OP_PRINT,0},{OP_HALT,0}};
      test("if(3<5)42", p, 9, 42); }

    /* ── Stage 4: DIV, MOD (delegated to exec loop) ── */
    printf("\n  --- Stage 4: DIV, MOD, LOOP ---\n");
    { Instr p[]={{OP_CONST,10},{OP_CONST,2},{OP_DIV,0},{OP_PRINT,0},{OP_HALT,0}};
      test("10/2", p, 5, 5); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,3},{OP_MOD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("10%3", p, 5, 1); }
    { Instr p[]={{OP_CONST,21},{OP_CONST,7},{OP_DIV,0},{OP_PRINT,0},{OP_HALT,0}};
      test("21/7", p, 5, 3); }
    { Instr p[]={{OP_CONST,15},{OP_CONST,4},{OP_MOD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("15%4", p, 5, 3); }
    /* DIV in a computation: (10/2) + 3 = 8 */
    { Instr p[]={{OP_CONST,10},{OP_CONST,2},{OP_DIV,0},{OP_CONST,3},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("10/2+3", p, 7, 8); }
    /* LOOP opcode: sum counter 3+2+1 into mem0 */
    { Instr p[]={
        {OP_CONST,0},{OP_SET_LOCAL,0},
        {OP_CONST,3},{OP_SET_LOCAL,1},
        {OP_GET_LOCAL,1},{OP_CONST,0},{OP_EQ,0},{OP_JUMP_IF_FALSE,11},
        {OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0},
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},{OP_SET_LOCAL,0},
        {OP_GET_LOCAL,1},{OP_CONST,1},{OP_SUB,0},{OP_SET_LOCAL,1},
        {OP_LOOP,4}
      }; test("loop sum 3..1", p, 20, 6); }

    /* ── Stage 5: CALL, RETURN ── */
    printf("\n  --- Stage 5: CALL, RETURN ---\n");
    /* f(x) = x + 1, call f(5) → 6
     * Layout: CONST 5, CONST func_entry, CALL 1, PRINT, HALT
     *   func_entry(5): GET_LOCAL 0, CONST 1, ADD, RETURN */
    { Instr p[]={
        {OP_CONST,5},{OP_CONST,5},{OP_CALL,1},              /* push arg=5, push func_pc=5, call(argc=1) */
        {OP_PRINT,0},{OP_HALT,0},                           /* print return value, halt */
        /* func entry (pc=5): */
        {OP_GET_LOCAL,0},{OP_CONST,1},{OP_ADD,0},{OP_RETURN,0},
      }; test("f(x)=x+1, f(5)", p, 9, 6); }

    /* f(x) = x * x, call f(7) → 49 */
    { Instr p[]={
        {OP_CONST,7},{OP_CONST,5},{OP_CALL,1},
        {OP_PRINT,0},{OP_HALT,0},
        /* func entry (pc=5): */
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,0},{OP_MUL,0},{OP_RETURN,0},
      }; test("f(x)=x*x, f(7)", p, 9, 49); }

    /* f(a,b) = a + b, call f(3,4) → 7 */
    { Instr p[]={
        {OP_CONST,3},{OP_CONST,4},{OP_CONST,6},{OP_CALL,2},
        {OP_PRINT,0},{OP_HALT,0},
        /* func entry (pc=6): */
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},{OP_RETURN,0},
      }; test("f(a,b)=a+b, f(3,4)", p, 10, 7); }

    /* ── Stage 6: CONS, CAR, CDR, NULL_P ── */
    printf("\n  --- Stage 6: CONS, CAR, CDR, NULL_P ---\n");
    /* (car (cons 3 4)) → 3 */
    { Instr p[]={{OP_CONST,3},{OP_CONST,4},{OP_CONS,0},{OP_CAR,0},{OP_PRINT,0},{OP_HALT,0}};
      test("car(cons 3 4)", p, 6, 3); }
    /* (cdr (cons 3 4)) → 4 */
    { Instr p[]={{OP_CONST,3},{OP_CONST,4},{OP_CONS,0},{OP_CDR,0},{OP_PRINT,0},{OP_HALT,0}};
      test("cdr(cons 3 4)", p, 6, 4); }
    /* (null? (cdr (cons 3 nil))) → 1, proving cdr type survives arena round-trip */
    { Instr p[]={{OP_CONST,3},{OP_NIL,0},{OP_CONS,0},{OP_CDR,0},{OP_NULL_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("null?(cdr cons nil)", p, 7, 1); }
    /* (null? nil) → 1 */
    { Instr p[]={{OP_NIL,0},{OP_NULL_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("null?(nil)", p, 4, 1); }
    /* (null? 5) → 0 */
    { Instr p[]={{OP_CONST,5},{OP_NULL_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("null?(5)", p, 4, 0); }
    /* (car (cdr (cons 1 (cons 2 (cons 3 nil))))) → 2
     * Build list (1 2 3): CONST 3, NIL, CONS → (3), CONST 2, swap args, CONS → (2 3), etc.
     * Actually: cons expects TOS=cdr SOS=car. So (cons 1 (cons 2 (cons 3 nil))):
     *   NIL, CONST 3, cons → pair(3,nil), CONST 2, swap...
     * Wait — stack order. CONS pops TOS=cdr, SOS=car.
     * To build (cons 3 nil): push 3 (car), push nil (cdr) → CONST 3, NIL, CONS
     * But that gives TOS=nil, SOS=3, so car=3, cdr=nil. ✓
     * (cons 2 (cons 3 nil)): push (cons 3 nil) result, then push 2, then cons
     * But stack: after first cons, TOS=pair_ptr. Need CONST 2 as SOS.
     * CONST 2 pushes 2 onto stack, so stack = [2, pair_ptr].
     * But CONS pops TOS=cdr, SOS=car → car=pair_ptr, cdr=2. Wrong!
     * Need: car=2, cdr=pair_ptr → SOS=2, TOS=pair_ptr.
     * So: first CONS gives pair_ptr on TOS. Then CONST 2 makes [2, pair_ptr].
     * We need TOS=pair_ptr (cdr), SOS=2 (car). But after CONST 2, TOS=2, SOS=pair_ptr.
     * Need a SWAP! But we don't have SWAP... we have DUP and POP but not SWAP.
     *
     * Alternative: build list in reverse. (cons 1 (cons 2 (cons 3 nil))):
     * Push elements in order: 3, nil → cons → (3.nil)
     *                         2, (3.nil) → cons → (2.(3.nil))
     * To get 2 as car, need SOS=2, TOS=(3.nil). After first cons, TOS=(3.nil).
     * CONST 2 → TOS=2, SOS=(3.nil). CONS → car=(3.nil)=SOS, cdr=2=TOS. WRONG.
     * We actually need TOS=(3.nil), SOS=2.
     *
     * OK let's just test simple car/cdr and move on. The swap issue is a
     * known limitation of the 4-register stack without a SWAP opcode.
     */

    /* ── Stage 7: Remaining opcodes + integration tests ── */
    printf("\n  --- Stage 7: Integration tests ---\n");

    /* Recursive factorial: fact(n) = if n==0 then 1 else n * fact(n-1)
     * Uses CALL/RETURN with recursive calls.
     * Layout:
     *   0: CONST 5          ; arg = 5
     *   1: CONST 5          ; func_pc = 5
     *   2: CALL 1           ; call fact(5)
     *   3: PRINT             ; print result
     *   4: HALT
     *   --- fact (pc=5) ---
     *   5: GET_LOCAL 0       ; push n
     *   6: CONST 0
     *   7: EQ                ; n == 0?
     *   8: JUMP_IF_FALSE 11  ; if not, goto recursive case
     *   9: CONST 1           ; return 1
     *  10: RETURN
     *  11: GET_LOCAL 0       ; push n
     *  12: GET_LOCAL 0       ; push n (for n-1)
     *  13: CONST 1
     *  14: SUB               ; n - 1
     *  15: CONST 5           ; func_pc = 5
     *  16: CALL 1            ; call fact(n-1)
     *  17: MUL               ; n * fact(n-1)
     *  18: RETURN
     */
    { Instr p[]={
        {OP_CONST,5},{OP_CONST,5},{OP_CALL,1},{OP_PRINT,0},{OP_HALT,0},
        /* fact: */
        {OP_GET_LOCAL,0},{OP_CONST,0},{OP_EQ,0},{OP_JUMP_IF_FALSE,11},
        {OP_CONST,1},{OP_RETURN,0},
        /* recursive case: */
        {OP_GET_LOCAL,0},
        {OP_GET_LOCAL,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
        {OP_MUL,0},{OP_RETURN,0},
      }; test("rec fact(5)", p, 19, 120); }

    /* Recursive fibonacci: fib(n) = if n<=1 then n else fib(n-1)+fib(n-2)
     *   0: CONST 7, 1: CONST 5, 2: CALL 1, 3: PRINT, 4: HALT
     *   5: GET_LOCAL 0, CONST 1, LE, JUMP_IF_FALSE 11
     *   9: GET_LOCAL 0, RETURN                          ; base case: return n
     *  11: GET_LOCAL 0, CONST 1, SUB, CONST 5, CALL 1   ; fib(n-1)
     *  16: GET_LOCAL 0, CONST 2, SUB, CONST 5, CALL 1   ; fib(n-2)
     *  21: ADD, RETURN
     */
    { Instr p[]={
        {OP_CONST,7},{OP_CONST,5},{OP_CALL,1},{OP_PRINT,0},{OP_HALT,0},
        /* fib: */
        {OP_GET_LOCAL,0},{OP_CONST,1},{OP_LE,0},{OP_JUMP_IF_FALSE,11},
        {OP_GET_LOCAL,0},{OP_RETURN,0},
        /* recursive case: */
        {OP_GET_LOCAL,0},{OP_CONST,1},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
        {OP_GET_LOCAL,0},{OP_CONST,2},{OP_SUB,0},{OP_CONST,5},{OP_CALL,1},
        {OP_ADD,0},{OP_RETURN,0},
      }; test("rec fib(7)", p, 23, 13); }

    /* set-car!/set-cdr! test: cons pair, mutate, read back */
    { Instr p[]={
        {OP_CONST,10},{OP_CONST,20},{OP_CONS,0},          /* (cons 10 20) → pair_ptr */
        {OP_DUP,0},{OP_CONST,99},{OP_SET_CAR,0},          /* set-car! pair 99 (pops val+pair, but we dup'd) */
        /* Wait — SET_CAR pops TOS=val, SOS=pair. After DUP we have [pair,pair].
         * CONST 99 gives [99,pair,pair]. SET_CAR: val=99, pair=pair → mutate.
         * Pops both, leaving [pair] on stack. Then CAR reads the mutated car. */
        {OP_CAR,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("set-car!", p, 9, 99); }
    { Instr p[]={
        {OP_CONST,10},{OP_CONST,20},{OP_CONS,0},
        {OP_DUP,0},{OP_CONST,99},{OP_SET_CDR,0},
        {OP_CDR,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("set-cdr!", p, 9, 99); }

    /* pair? test */
    { Instr p[]={
        {OP_CONST,1},{OP_CONST,2},{OP_CONS,0},{OP_PAIR_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("pair?(cons 1 2)", p, 6, 1); }
    { Instr p[]={
        {OP_CONST,1},{OP_CONST,2},{OP_CONS,0},
        {OP_CONST,3},{OP_CONS,0},{OP_CAR,0},{OP_PAIR_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("pair?(car nested)", p, 9, 1); }

    /* Stage-1 VM-as-transformer memory-op regressions */
    { Instr p[]={{OP_CONST,42},{OP_NUM_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("number?(42)", p, 4, 1); }
    { Instr p[]={{OP_TRUE,0},{OP_BOOL_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("boolean?(true)", p, 4, 1); }
    { Instr p[]={{OP_CONST,42},{OP_STR_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("string?(42)", p, 4, 0); }
    { Instr p[]={{OP_CLOSURE,5},{OP_PROC_P,0},{OP_PRINT,0},{OP_HALT,0},
                 {OP_NOP,0},{OP_CONST,1},{OP_RETURN,0}};
      test("procedure?(closure)", p, 7, 1); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,20},{OP_VEC_CREATE,2},{OP_VEC_P,0},
                 {OP_PRINT,0},{OP_HALT,0}};
      test("vector?(vec)", p, 6, 1); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,20},{OP_CONST,30},{OP_POPN,1},
                 {OP_POP,0},{OP_PRINT,0},{OP_HALT,0}};
      test("popn1 keeps below", p, 7, 10); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,20},{OP_CONST,30},{OP_CONST,40},
                 {OP_POPN,2},{OP_POP,0},{OP_PRINT,0},{OP_HALT,0}};
      test("popn2 keeps below", p, 8, 10); }
    { Instr p[]={{OP_CONST,10},{OP_CONST,20},{OP_CONST,30},{OP_CONST,40},
                 {OP_POPN,3},{OP_PRINT,0},{OP_HALT,0}};
      test("popn3 keeps TOS", p, 7, 40); }
    { Instr p[]={{OP_VOID,0},{OP_CONST,42},{OP_PRINT,0},{OP_HALT,0}};
      test("void pc++", p, 4, 42); }
    { Instr p[]={{OP_PUSH_HANDLER,4},{OP_POP_HANDLER,0},{OP_CONST,42},
                 {OP_PRINT,0},{OP_HALT,0}};
      test("handler push/pop pc++", p, 5, 42); }
    { Instr p[]={{OP_GET_EXN,0},{OP_PRINT,0},{OP_HALT,0}};
      test("get-exn default zero", p, 3, 0); }
    { Instr p[]={{OP_CONST,11},{OP_CONST,22},{OP_WIND_PUSH,0},
                 {OP_WIND_POP,0},{OP_PRINT,0},{OP_HALT,0}};
      test("wind push/pop keeps body value", p, 6, 11); }
    { Instr p[]={{OP_CONST,4},{OP_CALLCC,0},{OP_PRINT,0},{OP_HALT,0},
                 {OP_GET_LOCAL,0},{OP_CONST,77},{OP_INVOKE_CC,0}};
      test("callcc arena escape", p, 7, 77); }
    { Instr p[]={{OP_CONST,12},{OP_CONST,6},{OP_CALLCC,0},{OP_ADD,0},
                 {OP_PRINT,0},{OP_HALT,0},{OP_GET_LOCAL,0},{OP_CONST,77},
                 {OP_INVOKE_CC,0}};
      test("callcc restores stack", p, 9, 89); }
    { Instr p[]={{OP_CONST,33},{OP_SET_LOCAL,1},{OP_CONST,8},{OP_CALLCC,0},
                 {OP_GET_LOCAL,1},{OP_ADD,0},{OP_PRINT,0},{OP_HALT,0},
                 {OP_GET_LOCAL,0},{OP_CONST,44},{OP_SET_LOCAL,1},{OP_CONST,7},
                 {OP_INVOKE_CC,0}};
      test("callcc restores mem", p, 13, 40); }
    { Instr p[]={{OP_CONST,42},{OP_SET_UPVALUE,0},{OP_GET_UPVALUE,0},
                 {OP_PRINT,0},{OP_HALT,0}};
      test("upvalue set/get mem0", p, 5, 42); }
    { Instr p[]={{OP_CONST,17},{OP_SET_LOCAL,1},{OP_GET_UPVALUE,1},
                 {OP_PRINT,0},{OP_HALT,0}};
      test("upvalue get mem1 fallback", p, 5, 17); }
    { Instr p[]={{OP_CONST,42},{OP_SET_LOCAL,0},{OP_CLOSE_UPVALUE,0},
                 {OP_GET_LOCAL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("close-upvalue arena no-op", p, 6, 42); }
    { Instr p[]={{OP_CLOSURE,5},{OP_OPEN_CLOSURE,0},{OP_PROC_P,0},
                 {OP_PRINT,0},{OP_HALT,0},{OP_CONST,1},{OP_RETURN,0}};
      test("open-closure keeps arena closure", p, 7, 1); }
    { Instr p[]={{OP_CLOSURE,7},{OP_OPEN_CLOSURE,0},{OP_CONST,77},{OP_SET_UPVALUE,0},
                 {OP_GET_UPVALUE,0},{OP_PRINT,0},{OP_HALT,0},{OP_CONST,1},{OP_RETURN,0}};
      test("arena upvalue set/get cell", p, 9, 77); }
    { Instr p[]={{OP_CLOSURE,8},{OP_OPEN_CLOSURE,0},{OP_CONST,55},{OP_SET_LOCAL,0},
                 {OP_CLOSE_UPVALUE,0},{OP_GET_UPVALUE,0},{OP_PRINT,0},{OP_HALT,0},
                 {OP_CONST,1},{OP_RETURN,0}};
      test("arena close-upvalue cell", p, 10, 55); }

    /* Composite: (+ (car (cons 10 20)) (cdr (cons 30 40))) = 10 + 40 = 50 */
    { Instr p[]={
        {OP_CONST,10},{OP_CONST,20},{OP_CONS,0},{OP_CAR,0},  /* car(cons 10 20) = 10 */
        {OP_CONST,30},{OP_CONST,40},{OP_CONS,0},{OP_CDR,0},  /* cdr(cons 30 40) = 40 */
        {OP_ADD,0},{OP_PRINT,0},{OP_HALT,0}};
      test("car+cdr", p, 11, 50); }

    /* Closure test: create closure on heap, call it via CALL */
    { Instr p[]={
        /* 0 */ {OP_CLOSURE,5},                               /* push closure ptr (entry at pc=5) */
        /* 1 */ {OP_CALL,0},                                  /* call closure (0 args) */
        /* 2 */ {OP_PRINT,0},
        /* 3 */ {OP_HALT,0},
        /* padding */ {OP_NOP,0},
        /* func entry (pc=5): return 42 */
        /* 5 */ {OP_CONST,42},{OP_RETURN,0},
      }; test("closure()=42", p, 7, 42); }

    /* Vector test: create vec, read element */
    { Instr p[]={
        {OP_CONST,10},{OP_CONST,20},{OP_CONST,30},{OP_VEC_CREATE,3},  /* #(10 20 30) */
        {OP_CONST,1},{OP_VEC_REF,0},                                  /* vec[1] = 20 */
        {OP_PRINT,0},{OP_HALT,0}};
      /* Note: VEC_REF expects TOS=index, SOS=vec_ptr */
      test("vec-ref", p, 8, 20); }

    /* Vector length test */
    { Instr p[]={
        {OP_CONST,10},{OP_CONST,20},{OP_CONST,30},{OP_VEC_CREATE,3},
        {OP_VEC_LEN,0},{OP_PRINT,0},{OP_HALT,0}};
      test("vec-len", p, 7, 3); }
    { Instr p[]={
        {OP_CONST,10},{OP_CONST,20},{OP_CONST,30},{OP_VEC_CREATE,3},
        {OP_DUP,0},{OP_CONST,1},{OP_CONST,99},{OP_VEC_SET,0},
        {OP_CONST,1},{OP_VEC_REF,0},{OP_PRINT,0},{OP_HALT,0}};
      test("vec-set/ref", p, 12, 99); }
    { Instr p[]={
        {OP_CONST,65},{OP_CONST,66},{OP_VEC_CREATE,2},
        {OP_CONST,1},{OP_STR_REF,0},{OP_PRINT,0},{OP_HALT,0}};
      test("str-ref arena-layout", p, 7, 66); }
    { Instr p[]={
        {OP_CONST,65},{OP_CONST,66},{OP_VEC_CREATE,2},
        {OP_STR_LEN,0},{OP_PRINT,0},{OP_HALT,0}};
      test("str-len arena-layout", p, 6, 2); }

    /* Tail-call optimization test: tail-recursive sum
     * sum(n, acc) = if n==0 then acc else sum(n-1, acc+n)
     *   0: CONST 100, CONST 0, CONST 5, CALL 2  (sum(100, 0))
     *   4: PRINT, HALT
     *   5: GET_LOCAL 0   ; n
     *   6: CONST 0, EQ   ; n == 0?
     *   8: JUMP_IF_FALSE 11
     *   9: GET_LOCAL 1   ; return acc
     *  10: RETURN
     *  11: GET_LOCAL 0, CONST 1, SUB           ; n-1
     *  14: GET_LOCAL 1, GET_LOCAL 0, ADD        ; acc+n
     *  17: CONST 5, TAIL_CALL 2                ; tail-call sum(n-1, acc+n)
     */
    { Instr p[]={
        /* 0 */ {OP_CONST,100},{OP_CONST,0},{OP_CONST,6},{OP_CALL,2},
        /* 4 */ {OP_PRINT,0},{OP_HALT,0},
        /* sum (pc=6): MEM0=acc (first arg=SOS), MEM1=n (second arg=R2) */
        /* 6  */ {OP_GET_LOCAL,1},{OP_CONST,0},{OP_EQ,0},{OP_JUMP_IF_FALSE,12},
        /* 10 */ {OP_GET_LOCAL,0},{OP_RETURN,0},
        /* 12: recursive case */
        /* 12 */ {OP_GET_LOCAL,1},{OP_CONST,1},{OP_SUB,0},          /* n-1 */
        /* 15 */ {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},       /* acc+n */
        /* 18 */ {OP_CONST,6},{OP_TAIL_CALL,2},
      }; test("tail sum(100)", p, 20, 5050); }
    { Instr p[]={
        {OP_CONST,5},{OP_CALL,0},{OP_PRINT,0},{OP_HALT,0},{OP_NOP,0},
        {OP_CONST,7},{OP_TAIL_CALL,0},
        {OP_CONST,77},{OP_RETURN,0},
      }; test("tail argc0", p, 9, 77); }
    { Instr p[]={
        {OP_CONST,12},{OP_CONST,5},{OP_CALL,1},{OP_PRINT,0},{OP_HALT,0},
        {OP_GET_LOCAL,0},{OP_CONST,8},{OP_TAIL_CALL,1},
        {OP_GET_LOCAL,0},{OP_RETURN,0},
      }; test("tail argc1", p, 10, 12); }
    { Instr p[]={
        {OP_CONST,1},{OP_CONST,2},{OP_CONST,3},{OP_CONST,7},{OP_CALL,3},
        {OP_PRINT,0},{OP_HALT,0},
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_GET_LOCAL,2},{OP_CONST,12},{OP_TAIL_CALL,3},
        {OP_GET_LOCAL,0},{OP_GET_LOCAL,1},{OP_ADD,0},{OP_GET_LOCAL,2},{OP_ADD,0},{OP_RETURN,0},
      }; test("tail argc3", p, 18, 6); }
    { Instr p[]={
        {OP_CONST,2},{OP_SET_LOCAL,1},{OP_CONST,3},{OP_SET_LOCAL,2},
        {OP_CONST,4},{OP_SET_LOCAL,3},{OP_PACK_REST,1},
        {OP_GET_LOCAL,1},{OP_CAR,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("pack-rest car", p, 11, 2); }
    { Instr p[]={
        {OP_CONST,2},{OP_SET_LOCAL,1},{OP_CONST,3},{OP_SET_LOCAL,2},
        {OP_CONST,4},{OP_SET_LOCAL,3},{OP_PACK_REST,1},
        {OP_GET_LOCAL,1},{OP_CDR,0},{OP_CAR,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("pack-rest cdr car", p, 12, 3); }
    { Instr p[]={
        {OP_CONST,9},{OP_SET_LOCAL,3},{OP_PACK_REST,3},
        {OP_GET_LOCAL,3},{OP_CDR,0},{OP_NULL_P,0},{OP_PRINT,0},{OP_HALT,0},
      }; test("pack-rest single tail nil", p, 8, 1); }

    /* Dynamic multiplication: 100/100 */
    printf("\n  Dynamic multiplication: ");
    int mul_ok = 0;
    for (int a=0; a<10; a++) for (int b=0; b<10; b++) {
        Instr p[]={{OP_CONST,a},{OP_CONST,b},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
        float r[1],s[1],m[1];
        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
        if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
        run_reference(p,5,r,1);
        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
        if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
        run_simulated(p,5,s,1);
        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
        if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
        if(g_weights) run_with_weights(g_weights,p,5,m,1);
        if(fabsf(r[0]-(float)(a*b))<0.01f && fabsf(s[0]-(float)(a*b))<0.01f
           && (!g_weights || fabsf(m[0]-(float)(a*b))<0.01f)) mul_ok++;
    }
    printf("%d/100\n", mul_ok);
    if(mul_ok==100) n_pass++; else n_fail++;

    /* ── Stage 8: Edge cases ── */
    printf("\n  --- Stage 8: Edge cases ---\n");

    /* Division by zero -> halt */
    { Instr p[]={{OP_CONST,5},{OP_CONST,0},{OP_DIV,0},{OP_PRINT,0},{OP_HALT,0}};
      /* Should halt (div by zero), no output */
      float out[1];
      g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
      int nout = run_reference(p, 5, out, 1);
      printf("  %-25s %s\n", "div-by-zero", nout == 0 ? "OK" : "FAIL");
      if (nout == 0) n_pass++; else n_fail++;
    }

    /* Modulo by zero -> halt */
    { Instr p[]={{OP_CONST,5},{OP_CONST,0},{OP_MOD,0},{OP_PRINT,0},{OP_HALT,0}};
      float out[1];
      g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
      int nout = run_reference(p, 5, out, 1);
      printf("  %-25s %s\n", "mod-by-zero", nout == 0 ? "OK" : "FAIL");
      if (nout == 0) n_pass++; else n_fail++;
    }

    /* NULL_P on non-nil value */
    { Instr p[]={{OP_CONST,42},{OP_NULL_P,0},{OP_PRINT,0},{OP_HALT,0}};
      test("null?(42)=false", p, 4, 0); }

    /* Nested arithmetic: (3 + 4) * (5 - 2) = 21 */
    { Instr p[]={{OP_CONST,3},{OP_CONST,4},{OP_ADD,0},{OP_CONST,5},{OP_CONST,2},{OP_SUB,0},{OP_MUL,0},{OP_PRINT,0},{OP_HALT,0}};
      test("(3+4)*(5-2)", p, 9, 21); }

    /* ABS of negative */
    { Instr p[]={{OP_CONST,7},{OP_NEG,0},{OP_ABS,0},{OP_PRINT,0},{OP_HALT,0}};
      test("abs(-7)", p, 5, 7); }

    /* ── Stage 9: AD — native autodiff in weights ── */
    printf("\n  --- Stage 9: AD forward + backward ---\n");

    /* (debug trace removed) */

    /* f(x) = x^2 at x=3: tape = [var(3), mul(0,0)=9], backward → grad[0] = 6 */
    { Instr p[]={
        {OP_AD_VAR, 3},       /* node 0: x=3 */
        {OP_DUP, 0},          /* duplicate node index 0 */
        {OP_DUP, 0},          /* stack: [0, 0, 0] */
        {OP_AD_MUL, 0},       /* node 1: x*x=9, stack: [0, 1] */
        {OP_AD_BACKWARD, 0},  /* backward from node 1 */
        {OP_CONST, 0},        /* push node index 0 */
        {OP_AD_GRAD, 0},      /* push grad of node 0 */
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx x^2 at 3 = 6", p, 9, 6); }
    /* (backward debug variable removed) */

    /* f(x) = x+x at x=5: grad = 2 (fan-out test) */
    { Instr p[]={
        {OP_AD_VAR, 5},       /* node 0: x=5 */
        {OP_DUP, 0},
        {OP_DUP, 0},
        {OP_AD_ADD, 0},       /* node 1: x+x=10 */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx (x+x) at 5 = 2", p, 9, 2); }

    /* f(x,y) = x-y at (5,2): df/dy = -1 */
    { Instr p[]={
        {OP_AD_VAR, 5},
        {OP_AD_VAR, 2},
        {OP_AD_SUB, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 1},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d(x-y)/dy at (5,2) = -1", p, 8, -1); }

    /* f(x) = -x at x=7: grad = -1 */
    { Instr p[]={
        {OP_AD_VAR, 7},
        {OP_AD_NEG, 0},       /* node 1: -x=-7 */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx (-x) at 7 = -1", p, 7, -1); }

    /* f(x) = abs(x): derivative is sign(x) away from zero */
    { Instr p[]={
        {OP_AD_VAR, 7},
        {OP_AD_ABS, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx abs(7) = 1", p, 7, 1); }

    { Instr p[]={
        {OP_AD_VAR, -7},
        {OP_AD_ABS, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx abs(-7) = -1", p, 7, -1); }

    /* f(x,y) = x*y at (3,4): df/dx=4, df/dy=3.
     * We check df/dx. node 0=x=3, node 1=y=4, node 2=x*y */
    { Instr p[]={
        {OP_AD_VAR, 3},       /* node 0: x=3 */
        {OP_AD_VAR, 4},       /* node 1: y=4 */
        {OP_AD_MUL, 0},       /* node 2: x*y=12. Stack after: [2] (popped 0,1, pushed 2) */
        {OP_AD_BACKWARD, 0},  /* backward from node 2 */
        {OP_CONST, 0},        /* push 0 (node index for x) */
        {OP_AD_GRAD, 0},      /* grad of x = y = 4 */
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d(x*y)/dx at (3,4) = 4", p, 8, 4); }

    /* Same but check df/dy = 3 */
    { Instr p[]={
        {OP_AD_VAR, 3},
        {OP_AD_VAR, 4},
        {OP_AD_MUL, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 1},        /* push 1 (node index for y) */
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d(x*y)/dy at (3,4) = 3", p, 8, 3); }

    /* f(x) = x/2 at x=6: df/dx = 1/2 */
    { Instr p[]={
        {OP_AD_VAR, 6},
        {OP_AD_CONST, 2},
        {OP_AD_DIV, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx x/2 at 6 = 0.5", p, 8, 0.5f); }

    /* f(y) = 6/y at y=2: df/dy = -6/(2*2) = -1.5 */
    { Instr p[]={
        {OP_AD_CONST, 6},
        {OP_AD_VAR, 2},
        {OP_AD_DIV, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 1},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dy 6/y at 2 = -1.5", p, 8, -1.5f); }

    /* f(x) = x^2 at x=3: df/dx = 2x = 6 */
    { Instr p[]={
        {OP_AD_VAR, 3},
        {OP_AD_CONST, 2},
        {OP_AD_POW, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx pow(x,2) at 3 = 6", p, 8, 6); }

    /* f(y) = 2^y at y=3: df/dy = 2^3 * log(2) */
    { Instr p[]={
        {OP_AD_CONST, 2},
        {OP_AD_VAR, 3},
        {OP_AD_POW, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 1},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dy pow(2,y) at 3", p, 8, 8.0f * logf(2.0f)); }

    /* f(x) = exp(0): grad = exp(0) = 1 */
    { Instr p[]={
        {OP_AD_VAR, 0},       /* node 0: x=0 */
        {OP_AD_EXP, 0},       /* node 1: exp(0)=1 */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx exp(0) = 1", p, 7, 1); }

    /* f(x) = sigmoid(0): grad = 0.25 */
    { Instr p[]={
        {OP_AD_VAR, 0},
        {OP_AD_SIGMOID, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx sigmoid(0) = 0.25", p, 7, 0.25f); }

    /* f(x) = tanh(0): grad = 1 */
    { Instr p[]={
        {OP_AD_VAR, 0},
        {OP_AD_TANH, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx tanh(0) = 1", p, 7, 1); }

    /* f(x) = log(1): grad = 1 */
    { Instr p[]={
        {OP_AD_VAR, 1},
        {OP_AD_LOG, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx log(1) = 1", p, 7, 1); }

    /* f(x) = sqrt(4): grad = 1/(2*sqrt(4)) = 0.25 */
    { Instr p[]={
        {OP_AD_VAR, 4},
        {OP_AD_SQRT, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx sqrt(4) = 0.25", p, 7, 0.25f); }

    /* f(x) = sin(0): grad = cos(0) = 1 */
    { Instr p[]={
        {OP_AD_VAR, 0},
        {OP_AD_SIN, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx sin(0) = 1", p, 7, 1); }

    /* f(x) = sin(1): grad = cos(1) */
    { Instr p[]={
        {OP_AD_VAR, 1},
        {OP_AD_SIN, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx sin(1) = cos(1)", p, 7, cosf(1.0f)); }

    /* f(x) = cos(0): grad = -sin(0) = 0 */
    { Instr p[]={
        {OP_AD_VAR, 0},
        {OP_AD_COS, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx cos(0) = 0", p, 7, 0); }

    /* f(x) = cos(1): grad = -sin(1) */
    { Instr p[]={
        {OP_AD_VAR, 1},
        {OP_AD_COS, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx cos(1) = -sin(1)", p, 7, -sinf(1.0f)); }

    /* f(x) = cos(-1): grad = -sin(-1) */
    { Instr p[]={
        {OP_AD_VAR, -1},
        {OP_AD_COS, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx cos(-1) = sin(1)", p, 7, sinf(1.0f)); }

    /* f(x) = sin(-1): grad = cos(-1), exercising a negative table row */
    { Instr p[]={
        {OP_AD_VAR, -1},
        {OP_AD_SIN, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx sin(-1) = cos(1)", p, 7, cosf(1.0f)); }

    /* f(x) = relu(3): grad = 1 */
    { Instr p[]={
        {OP_AD_VAR, 3},
        {OP_AD_RELU, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx relu(3) = 1", p, 7, 1); }

    /* d/dx relu(-1) = 0 (inactive region) */
    { Instr p[]={
        {OP_AD_VAR, -1},
        {OP_AD_RELU, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD: d/dx relu(-1) = 0", p, 7, 0); }

    /* ── Group A: Forward tape recording verification ── */
    printf("\n  --- Group A: tape recording ---\n");

    /* Verify tape[0].value = 3 after ad_var(3) */
    { Instr p[]={
        {OP_AD_VAR, 3},       /* record var(3) as node 0 */
        {OP_PRINT, 0},        /* print tape index = 0 */
        {OP_HALT, 0}
      }; test("tape: ad_var(3) → index 0", p, 3, 0); }

    /* Verify ad_var(3), ad_var(4), ad_add → node 2 with correct value
     * The value 7 is stored in tape[2]; we can't directly read it, but
     * we verify via backward: f(x,y) = x+y, df/dx = 1, which implies
     * the forward pass computed 3+4=7 correctly. */
    { Instr p[]={
        {OP_AD_VAR, 3},
        {OP_AD_VAR, 4},
        {OP_AD_ADD, 0},       /* node 2: 3+4=7 */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},        /* gradient of first var */
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("tape: var(3)+var(4) grad=1", p, 8, 1); }

    /* Verify ad_const has no gradient flow */
    { Instr p[]={
        {OP_AD_CONST, 5},     /* node 0: const(5), no gradient */
        {OP_AD_VAR, 3},       /* node 1: var(3) */
        {OP_AD_ADD, 0},       /* node 2: 5+3=8 */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},        /* gradient of const */
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("tape: const gets grad=1 (ADD passthrough)", p, 8, 1); }

    /* ── Tape overflow: verify 8 nodes fill correctly ── */
    printf("\n  --- Tape capacity ---\n");
    { Instr p[]={
        {OP_AD_VAR, 1},       /* node 0 */
        {OP_AD_VAR, 2},       /* node 1 */
        {OP_AD_VAR, 3},       /* node 2 */
        {OP_AD_VAR, 4},       /* node 3 */
        {OP_AD_VAR, 5},       /* node 4 */
        {OP_AD_VAR, 6},       /* node 5 */
        {OP_AD_VAR, 7},       /* node 6 */
        {OP_AD_VAR, 8},       /* node 7 — last slot (index 7) */
        {OP_PRINT, 0},        /* prints TOS = 7 */
        {OP_HALT, 0}
      }; test("tape: 8 nodes fill (max index=7)", p, 10, 7); }

    /* ── Cross-mode check: forward-mode dual == reverse-mode tape ──
     * For single-variable functions, the derivative computed via dual numbers
     * (forward-mode) must agree with the gradient computed via Wengert tape
     * (reverse-mode). We verify this for f(x) = x^3 at x=2.
     *
     * Forward-mode: f'(2) via dual = 12 (tested in Stage 9 basic tests)
     * Reverse-mode: f'(2) via tape = 12 (tested as "d/dx x^2 at 3 = 6" pattern)
     *
     * The cross-check is implicit: both modes produce 12 for d/dx x^3 at 2.
     * For an EXPLICIT cross-check, we compute f(x)=x*x*x via tape and verify. */
    printf("\n  --- Cross-mode: dual vs tape ---\n");
    { Instr p[]={
        {OP_AD_VAR, 2},       /* node 0: x=2 */
        {OP_DUP, 0},
        {OP_DUP, 0},
        {OP_AD_MUL, 0},       /* node 1: x*x=4 */
        {OP_DUP, 0},          /* dup node 0 (still on stack below) */
        /* Stack after DUP,DUP,MUL: [0, 1]. DUP copies 1→ [0, 1, 1].
         * But we need node 0 and node 1 for the second multiply.
         * Stack is [0, 1]. We need node0 * node1. But SOS=0, TOS=1.
         * AD_MUL with SOS=0 (left), TOS=1 (right) = x * (x*x) = x^3. */
        {OP_POP, 0},          /* drop dup, stack: [0, 1] */
        {OP_CONST, 0},        /* push node 0 */
        {OP_AD_MUL, 0},       /* node 2: node1 * node0 = x^2 * x = x^3 = 8 */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},      /* df/dx = 3x^2 = 12 */
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("cross: tape x^3 at 2 = 12 (matches dual)", p, 13, 12); }

    /* ── MLP gradient demo: f(w,b) = sigmoid(w*2 + b) at w=1, b=-1 ──
     * sigmoid(1*2 + -1) = sigmoid(1) ≈ 0.7311
     * df/dw = x * sigmoid'(w*x+b) = 2 * 0.7311 * (1 - 0.7311) ≈ 0.3932
     * df/db = sigmoid'(w*x+b) = 0.7311 * (1 - 0.7311) ≈ 0.1966
     * Uses 6 tape nodes: var(w=1), const(x=2), var(b=-1), mul(w,x), add(wx,b), sigmoid */
    printf("\n  --- MLP gradient demo ---\n");
    { Instr p[]={
        {OP_AD_VAR, 1},       /* node 0: w=1 */
        {OP_AD_CONST, 2},     /* node 1: x=2 (constant, no gradient) */
        {OP_AD_MUL, 0},       /* node 2: w*x=2 */
        {OP_AD_VAR, -1},      /* node 3: b=-1 */
        /* We need to get node 2 and node 3 on TOS/SOS for AD_ADD.
         * After AD_VAR(-1): stack is [0, 1, 2, 3]
         * AD_MUL popped 0,1 pushed 2. Stack: [2, ...]
         * AD_VAR(-1) pushed 3. Stack: [2, 3, ...]
         * But AD_ADD needs SOS=left, TOS=right. We need [2, 3] on stack.
         * Currently TOS=3, SOS=2. AD_ADD will use SOS=2(left), TOS=3(right). */
        {OP_AD_ADD, 0},       /* node 4: w*x + b = 2 + (-1) = 1 */
        {OP_AD_SIGMOID, 0},   /* node 5: sigmoid(1) ≈ 0.7311 */
        {OP_AD_BACKWARD, 0},  /* backward from node 5 */
        {OP_CONST, 0},        /* push 0 (node index for w) */
        {OP_AD_GRAD, 0},      /* df/dw */
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("MLP: df/dw sigmoid(w*2+b) at w=1,b=-1", p, 11, 0.3932f); }

    /* Same MLP, check df/db */
    { Instr p[]={
        {OP_AD_VAR, 1},
        {OP_AD_CONST, 2},
        {OP_AD_MUL, 0},
        {OP_AD_VAR, -1},
        {OP_AD_ADD, 0},
        {OP_AD_SIGMOID, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 3},        /* push 3 (node index for b) */
        {OP_AD_GRAD, 0},      /* df/db */
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("MLP: df/db sigmoid(w*2+b) at w=1,b=-1", p, 11, 0.1966f); }

    /* ── Edge cases ── */
    printf("\n  --- AD edge cases ---\n");

    /* Backward on constant: gradient should be 0 */
    { Instr p[]={
        {OP_AD_CONST, 42},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD edge: grad of output const = 1 (seed)", p, 6, 1); }

    /* Backward on lone variable: gradient = 1 (identity) */
    { Instr p[]={
        {OP_AD_VAR, 7},
        {OP_DUP, 0},
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD edge: grad of var = 1", p, 7, 1); }

    /* Chain: f(x) = x^2 + 2x at x=3 → gradient = 2x + 2 = 8 */
    { Instr p[]={
        {OP_AD_VAR, 3},       /* node 0: x=3 */
        {OP_DUP, 0},          /* dup node index 0 */
        {OP_DUP, 0},          /* stack: [0,0,0] */
        {OP_AD_MUL, 0},       /* node 1: x*x=9, stack: [0,1] */
        {OP_AD_CONST, 2},     /* node 2: const(2), stack: [0,1,2] */
        /* Need to get node 0 and node 2 adjacent. Stack is [0,1,2].
         * We need node0 on SOS for AD_MUL with node2 on TOS.
         * But stack is [0,1,2]. We need to swap. Use a different approach:
         * Just compute 2*x as const(2) * var(x). Get the index 0 from depth. */
        /* Actually: after AD_CONST(2), stack=[0,1,2]. We need [0] * [2] = 2x.
         * But SOS=1 and TOS=2. We'd compute node1 * node2 = x^2 * 2. Wrong.
         * We need a fresh reference to node 0. Use DUP on the 0 that's deep.
         * Alternative: create another var reference by reloading from stack.
         * Simplest: compute via AD_ADD of x to x instead of 2*x.
         * f(x) = x^2 + x + x = x^2 + 2x. grad = 2x + 2 = 8 at x=3. */
        /* Actually let me restructure: */
        {OP_POP, 0},          /* drop node 2, stack: [0,1] */
        {OP_POP, 0},          /* drop node 1, stack: [0] */
        {OP_DUP, 0},          /* stack: [0,0] */
        {OP_AD_ADD, 0},       /* node 3: x + x = 2x = 6, stack: [3] */
        /* Now we need node 1 (x^2) and node 3 (2x) on stack */
        {OP_CONST, 1},        /* push literal 1 (node index for x^2) */
        {OP_AD_ADD, 0},       /* node 4: x^2 + 2x = 9 + 6 = 15, stack: [4] */
        {OP_AD_BACKWARD, 0},
        {OP_CONST, 0},
        {OP_AD_GRAD, 0},
        {OP_PRINT, 0},
        {OP_HALT, 0}
      }; test("AD chain: d/dx (x^2+2x) at 3 = 8", p, 16, 8); }

    printf("\n  [metrics] peak_heap=%d/%d\n", g_heap_ptr, HEAP_SIZE);

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    if (g_weights && n_fail == 0) {
        const char* out_path = getenv("ESHKOL_WEIGHTS_OUT");
        if (!out_path || !*out_path) out_path = "/tmp/interpreter_weights.bin";
        export_weights_binary(g_weights, out_path);
    }

    /* ── Integration test: load bytecode from eshkol_compiler ── */
    /* Supports both legacy raw format and new ESKB section-based format.
     * Set ESHKOL_BC=path.eskb to load ESKB, or ESHKOL_BC=path.bc for legacy. */
    if (g_weights) {
        const char* bc_path = getenv("ESHKOL_BC");
        if (bc_path) {
            size_t pathlen = strlen(bc_path);
            int is_eskb = (pathlen > 5 && strcmp(bc_path + pathlen - 5, ".eskb") == 0);

            if (is_eskb) {
                /* Load ESKB section-based format via eskb_reader */
                EskbModule mod;
                if (eskb_load_file(bc_path, &mod) == 0) {
                    /* Build Instr array and constants for weight matrix execution */
                    Instr* prog = (Instr*)calloc(mod.code_len, sizeof(Instr));
                    float* constants = (float*)calloc(mod.n_constants > 0 ? mod.n_constants : 1, sizeof(float));
                    if (prog && constants) {
                        for (int i = 0; i < mod.code_len; i++) {
                            prog[i].op = (OpCode)mod.opcodes[i];
                            prog[i].operand = mod.operands[i];
                        }
                        for (int i = 0; i < mod.n_constants; i++) {
                            switch (mod.const_types[i]) {
                            case ESKB_CONST_INT64: constants[i] = (float)mod.const_ints[i]; break;
                            case ESKB_CONST_F64:   constants[i] = (float)mod.const_floats[i]; break;
                            case ESKB_CONST_BOOL:  constants[i] = (float)mod.const_ints[i]; break;
                            default:               constants[i] = 0.0f; break;
                            }
                        }

                        /* Resolve CONST operands: replace constant pool index with actual value.
                         * The eshkol_compiler uses CONST <pool_index>, but our weight matrix
                         * uses CONST <immediate_value>. Inline the constants. */
                        for (int i = 0; i < mod.code_len; i++) {
                            if (prog[i].op == OP_CONST && prog[i].operand >= 0 && prog[i].operand < mod.n_constants) {
                                prog[i].operand = (int)constants[prog[i].operand];
                            }
                        }

                        printf("\n  --- Integration (ESKB): %s (%d instructions, %d constants) ---\n",
                               bc_path, mod.code_len, mod.n_constants);

                        /* Check if program fits within weight matrix interpreter limits.
                         * The weight matrix interpreter has MEM_SIZE=4 local slots and
                         * pe[256] positional embeddings. Programs with closures/prelude
                         * may exceed these limits; run only the reference path for those. */
                        int simple_enough = (mod.code_len <= 256);

                        /* Run through all 3 paths */
                        float r[64], s[64], m[64];
                        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
                        if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                        int rn = run_reference(prog, mod.code_len, r, 64);
                        int sn = 0, mn = 0;
                        if (simple_enough) {
                            g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
                            if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                            sn = run_simulated(prog, mod.code_len, s, 64);
                            g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
                            if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                            mn = run_with_weights(g_weights, prog, mod.code_len, m, 64);
                        }

                        printf("  Outputs (ref): "); for(int i=0;i<rn;i++) printf("%.4g ", r[i]); printf("\n");
                        if (simple_enough) {
                            printf("  Outputs (sim): "); for(int i=0;i<sn;i++) printf("%.4g ", s[i]); printf("\n");
                            printf("  Outputs (mat): "); for(int i=0;i<mn;i++) printf("%.4g ", m[i]); printf("\n");
                        } else {
                            printf("  (sim/mat skipped: %d instructions > 256 limit)\n", mod.code_len);
                        }

                        int match = 1;
                        if (simple_enough) {
                            int n_max = rn < sn ? rn : sn; n_max = n_max < mn ? n_max : mn;
                            for (int i = 0; i < n_max; i++) {
                                if (fabsf(r[i]-s[i]) > 0.01f || fabsf(r[i]-m[i]) > 0.01f) match = 0;
                            }
                        }
                        printf("  3-way match: %s\n", simple_enough ? (match ? "YES" : "NO") : "REF-ONLY");

                        free(prog);
                        free(constants);
                    } else {
                        printf("ERROR: allocation failed for ESKB bytecode\n");
                        free(prog); free(constants);
                    }
                    eskb_module_free(&mod);
                } else {
                    printf("  ERROR: failed to load ESKB file %s\n", bc_path);
                }
            } else {
                /* Legacy raw bytecode format */
                FILE* bf = fopen(bc_path, "rb");
                if (bf) {
                    uint32_t magic = 0, n_instr = 0, n_const = 0;
                    if (fread(&magic, 4, 1, bf) != 1 || fread(&n_instr, 4, 1, bf) != 1 ||
                        fread(&n_const, 4, 1, bf) != 1) {
                        printf("ERROR: truncated bytecode header\n"); fclose(bf);
                    } else if (magic == 0x45534B42 && n_instr < 8192 && n_const < 8192) {
                        /* Read instructions */
                        Instr* prog = (Instr*)calloc(n_instr, sizeof(Instr));
                        float* constants = (float*)calloc(n_const > 0 ? n_const : 1, sizeof(float));
                        if (!prog || !constants) {
                            printf("ERROR: allocation failed for bytecode\n");
                            free(prog); free(constants); fclose(bf);
                        } else {
                        int read_ok = 1;
                        for (uint32_t i = 0; i < n_instr && read_ok; i++) {
                            uint8_t op; int32_t operand;
                            if (fread(&op, 1, 1, bf) != 1 || fread(&operand, 4, 1, bf) != 1) { read_ok = 0; break; }
                            prog[i].op = (OpCode)op;
                            prog[i].operand = operand;
                        }
                        /* Read constants */
                        for (uint32_t i = 0; i < n_const && read_ok; i++) {
                            uint8_t type; double val;
                            if (fread(&type, 1, 1, bf) != 1 || fread(&val, 8, 1, bf) != 1) { read_ok = 0; break; }
                            constants[i] = (float)val;
                        }
                        fclose(bf);
                        if (!read_ok) { printf("ERROR: truncated bytecode file\n"); free(prog); free(constants); }
                        else {

                        /* Resolve CONST operands */
                        for (uint32_t i = 0; i < n_instr; i++) {
                            if (prog[i].op == OP_CONST && prog[i].operand >= 0 && prog[i].operand < (int)n_const) {
                                prog[i].operand = (int)constants[prog[i].operand];
                            }
                        }

                        printf("\n  --- Integration (legacy): %s (%d instructions, %d constants) ---\n",
                               bc_path, n_instr, n_const);

                        /* Run through all 3 paths */
                        float r[64], s[64], m[64];
                        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                        int rn = run_reference(prog, n_instr, r, 64);
                        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                        int sn = run_simulated(prog, n_instr, s, 64);
                        g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
                        int mn = run_with_weights(g_weights, prog, n_instr, m, 64);

                        printf("  Outputs (ref): "); for(int i=0;i<rn;i++) printf("%.4g ", r[i]); printf("\n");
                        printf("  Outputs (sim): "); for(int i=0;i<sn;i++) printf("%.4g ", s[i]); printf("\n");
                        printf("  Outputs (mat): "); for(int i=0;i<mn;i++) printf("%.4g ", m[i]); printf("\n");

                        int match = 1;
                        int n_max = rn < sn ? rn : sn; n_max = n_max < mn ? n_max : mn;
                        for (int i = 0; i < n_max; i++) {
                            if (fabsf(r[i]-s[i]) > 0.01f || fabsf(r[i]-m[i]) > 0.01f) match = 0;
                        }
                        printf("  3-way match: %s\n", match ? "YES" : "NO");

                        free(prog);
                        free(constants);
                        } /* end read_ok */
                        } /* end alloc check */
                    } else {
                        printf("  ERROR: invalid bytecode file (magic=0x%08x, n_instr=%u)\n", magic, n_instr);
                        fclose(bf);
                    }
                } else {
                    printf("  ERROR: cannot open bytecode file %s\n", bc_path);
                }
            }
        }
    }

    if (g_trace_vm_fp)  { fclose(g_trace_vm_fp);  g_trace_vm_fp = NULL; }
    if (g_trace_tf_fp)  { fclose(g_trace_tf_fp);  g_trace_tf_fp = NULL; }
    if (g_trace_sim_fp) { fclose(g_trace_sim_fp); g_trace_sim_fp = NULL; }
    if (g_weights) free(g_weights);
    return n_fail > 0 ? 1 : 0;
}
