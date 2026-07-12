static void layer0_attention(const float x[D], const float pe[][D], int np, float out[D]) {
    memset(out, 0, D * sizeof(float));
    float T = SCALE;
    float scores[256]; float mx = -1e30f;
    for (int p = 0; p < np && p < 256; p++) {
        scores[p] = (x[S_PC]*T*pe[p][0] + T*pe[p][1]) / sqrtf(2.0f);
        if (scores[p] > mx) mx = scores[p];
    }
    float sum = 0;
    for (int p = 0; p < np; p++) { scores[p] = expf(scores[p]-mx); sum += scores[p]; }
    for (int p = 0; p < np; p++) scores[p] /= sum;
    float v0=0, v1=0;
    for (int p = 0; p < np; p++) { v0 += scores[p]*pe[p][S_OPCODE]; v1 += scores[p]*pe[p][S_OPERAND]; }
    out[S_OPCODE] = v0; out[S_OPERAND] = v1;
}

/** @brief Simulated Layer 1 (preprocessing gated FFN): resolves addresses,
 *         evaluates comparisons, and loads the AD cursor ahead of opcode
 *         dispatch. */
static void layer1_ffn(const float x[D], float out[D]) {
    memset(out, 0, D*sizeof(float));
    /* Original: TOS * SOS via SQUARE trick */
    float a = x[S_TOS], b = x[S_SOS];
    out[S_PRODUCT] = 0.5f*(a+b)*(a+b) - 0.5f*a*a - 0.5f*b*b;
    /* AD products via SQUARE trick (polarization identity) */
    float g = x[S_AD_CUR_GRAD], lv = x[S_AD_LEFT_VALUE], rv = x[S_AD_RIGHT_VALUE];
    float sv = x[S_AD_CUR_SAVED];
    out[S_AD_PROD_GRAD_LV] = 0.5f*(g+rv)*(g+rv) - 0.5f*g*g - 0.5f*rv*rv; /* g * rv (MUL dL) */
    out[S_AD_PROD_GRAD_RV] = 0.5f*(g+lv)*(g+lv) - 0.5f*g*g - 0.5f*lv*lv; /* g * lv (MUL dR) */
    out[S_AD_PROD_LR] = 0.5f*(lv+rv)*(lv+rv) - 0.5f*lv*lv - 0.5f*rv*rv; /* lv * rv (MUL forward) */
    out[S_AD_PROD_GRAD_SV] = 0.5f*(g+sv)*(g+sv) - 0.5f*g*g - 0.5f*sv*sv; /* g * saved (ALL unary) */
}

/** @brief Simulated Layer 2 (product precompute, SQUARE activation):
 *         computes TOS*SOS and AD forward/backward product terms via the
 *         polarization identity, plus AD tape parent loads. */
static void layer2_ffn(const float x[D], float out[D]) {
    memset(out, 0, D*sizeof(float));
    /* GET_LOCAL address resolution: indicator(OPERAND==a) * mem[a] → LOADVAL */
    for (int a = 0; a < MEM_SIZE; a++)
        out[S_LOADVAL] += indicator(x[S_OPERAND], (float)a) * x[S_MEM0+a];
    /* SET_LOCAL store deltas: indicator(OPERAND==a) * (TOS - mem[a]) → STORED0+a */
    for (int a = 0; a < MEM_SIZE; a++)
        out[S_STORED0+a] = indicator(x[S_OPERAND], (float)a) * (x[S_TOS] - x[S_MEM0+a]);
    /* JUMP_IF_FALSE precompute */
    float iz = indicator(x[S_TOS], 0.0f);
    out[S_ZOPER] = iz * x[S_OPERAND];
    out[S_ZPC1]  = iz * (x[S_PC] + 1.0f);
    /* Bounded DIV/MOD precompute. S_ZPC1 is otherwise only live for
     * JUMP_IF_FALSE when TOS is zero, so nonzero DIV/MOD can reuse it as a
     * scratch result lane that Layer 3 clears before the next traced state. */
    float div_op = indicator(x[S_OPCODE], OP_DIV);
    for (int d = 1; d <= DIV_WEIGHT_MAX_DENOM; d++)
        out[S_ZPC1] += div_op * indicator(x[S_TOS], (float)d) * (x[S_SOS] / (float)d);
    for (int d = 1; d <= DIV_WEIGHT_MAX_DENOM; d++)
        out[S_IS_NATIVE] += div_op * indicator(x[S_TOS], (float)d);
    float mod_op = indicator(x[S_OPCODE], OP_MOD);
    for (int d = 3; d <= 4; d++) {
        for (int v = 0; v <= MOD_WEIGHT_MAX_NUM; v++) {
            out[S_ZPC1] += mod_op * indicator(x[S_TOS], (float)d) *
                           indicator(x[S_SOS], (float)v) * (float)(v % d);
            out[S_IS_NATIVE] += mod_op * indicator(x[S_TOS], (float)d) *
                                indicator(x[S_SOS], (float)v);
        }
    }
    /* Comparison precomputes */
    out[S_CMP_EQ] = indicator(x[S_TOS] - x[S_SOS], 0.0f);
    out[S_CMP_LT] = sigmoidf(SCALE * (x[S_TOS] - x[S_SOS] - 0.5f));
    /* ABS precompute */
    out[S_ABS_DELTA] = sigmoidf(SCALE * (-x[S_TOS] - 0.5f)) * (-2.0f * x[S_TOS]);

    /* Stage-1 VM-as-transformer type predicate precompute.
     * Layer 3 cannot synthesize both opcode and type indicators inside a
     * single gated neuron because the up path is linear. Precompute the
     * type-side indicators here, then gate by opcode in Layer 3. */
    out[S_TYPE_IS_NUM]  = indicator(x[S_TYPE_TOS], TYPE_NUMBER);
    out[S_TYPE_IS_BOOL] = indicator(x[S_TYPE_TOS], TYPE_BOOL);
    out[S_TYPE_IS_PAIR] = indicator(x[S_TYPE_TOS], TYPE_PAIR);
    out[S_TYPE_IS_PROC] = indicator(x[S_TYPE_TOS], TYPE_CLOSURE);
    out[S_TYPE_IS_STR]  = indicator(x[S_TYPE_TOS], TYPE_STRING);
    out[S_TYPE_IS_VEC]  = indicator(x[S_TYPE_TOS], TYPE_VECTOR);
    out[S_TYPE_IS_NIL]  = indicator(x[S_TYPE_TOS], TYPE_NIL);
    out[S_AD_UNARY_ABS_ACTIVE] = indicator(x[S_OPCODE], OP_AD_ABS);
    out[S_AD_UNARY_RELU_ACTIVE] = indicator(x[S_OPCODE], OP_AD_RELU);

    /* ── AD tape random-access load (backward mode only) ──
     * Load node at AD_CURSOR into AD_CUR_* fields.
     * Only active during backward pass — during forward ops, layer3 sets AD_CUR_* directly. */
    float bw_active = sigmoidf(SCALE * (x[S_AD_IS_BACKWARD] - 0.5f));
    float cursor = x[S_AD_CURSOR];
    for (int i = 0; i < AD_MAX_TAPE; i++) {
        float ci = indicator(cursor, (float)i) * bw_active;
        out[S_AD_CUR_OP]    += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_OP];
        out[S_AD_CUR_VALUE]  += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
        out[S_AD_CUR_GRAD]   += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD];
        out[S_AD_CUR_LEFT]   += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_LEFT];
        out[S_AD_CUR_RIGHT]  += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_RIGHT];
        out[S_AD_CUR_SAVED]  += ci * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_SAVED];
    }
    /* AD forward parent loads. With preprocessing before the SQUARE/product
     * and dispatch layers, bounded unary/table paths can record tape values
     * without C postprocess. */
    float fw_binary = (indicator(x[S_OPCODE], OP_AD_ADD) +
                       indicator(x[S_OPCODE], OP_AD_SUB) +
                       indicator(x[S_OPCODE], OP_AD_MUL) +
                       indicator(x[S_OPCODE], OP_AD_DIV) +
                       indicator(x[S_OPCODE], OP_AD_POW)) *
                      (1.0f - bw_active);
    float fw_unary_bounded = (indicator(x[S_OPCODE], OP_AD_ABS) +
                              indicator(x[S_OPCODE], OP_AD_RELU) +
                              indicator(x[S_OPCODE], OP_AD_SIGMOID) +
                              indicator(x[S_OPCODE], OP_AD_TANH) +
                              indicator(x[S_OPCODE], OP_AD_EXP) +
                              indicator(x[S_OPCODE], OP_AD_LOG) +
                              indicator(x[S_OPCODE], OP_AD_SQRT) +
                              indicator(x[S_OPCODE], OP_AD_SIN) +
                              indicator(x[S_OPCODE], OP_AD_COS)) *
                             (1.0f - bw_active);
    for (int i = 0; i < AD_MAX_TAPE; i++) {
        float li = indicator(x[S_SOS], (float)i) * fw_binary;
        float ri = indicator(x[S_TOS], (float)i) * fw_binary;
        float ui = indicator(x[S_TOS], (float)i) * fw_unary_bounded;
        float value = x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
        out[S_AD_LEFT_VALUE]  += li * value;
        out[S_AD_RIGHT_VALUE] += ri * value;
        out[S_AD_LEFT_VALUE]  += ui * value;
    }
    /* Parent load moved to layer4_ffn (runs after preprocessing in backward sequence) */

    /* Cursor decrement: IS_BACKWARD → delta[CURSOR] = -1 */
    if (bw_active > 0.5f) {
        out[S_AD_CURSOR] += -1.0f;
        /* Completion check: AD_IS_BACKWARD must be cleared on the cycle that
         * processes the LAST node (cursor == 0 pre-decrement → post-decrement
         * cursor == -1). The reference VM (ad_backward_step) does this in one
         * cycle: it processes node, decrements cursor, then if (cursor - 1 < 0)
         * clears AD_IS_BACKWARD same step. Originally this used
         *     at_done = indicator(cursor, -1.0f)
         * which fires only the cycle AFTER cursor went negative — adding a
         * spurious extra backward cycle that diverges from the reference. */
        float at_done = indicator(cursor, 0.0f);
        out[S_AD_IS_BACKWARD] += -at_done * x[S_AD_IS_BACKWARD];
        out[S_AD_MODE] += -at_done * x[S_AD_MODE];
        /* Transient clear: zero cursor-loaded fields + Zone D scratch */
        for (int d = S_AD_CUR_OP; d <= S_AD_RIGHT_VALUE; d++)
            out[d] += -x[d];
        out[S_AD_IS_FORWARD] += -x[S_AD_IS_FORWARD];
        for (int d = S_AD_GRAD_ACCUM; d <= S_AD_SPARE8; d++)
            out[d] += -x[d];
    }
}

/** @brief Simulated Layer 3 (execution gated FFN): the main opcode
 *         dispatch layer, computing each opcode's effect (stack ops, arena
 *         ops, AD forward/backward rules) as a sum of indicator-gated
 *         weight contributions. Mirrors execute_step()'s switch, but as a
 *         differentiable weight computation instead of a C branch. */
static void layer3_ffn(const float x[D], float out[D]) {
    memset(out, 0, D*sizeof(float));
    float op=x[S_OPCODE], oper=x[S_OPERAND], tos=x[S_TOS], sos=x[S_SOS];
    float r2=x[S_R2], r3=x[S_R3], product=x[S_PRODUCT], lv=x[S_LOADVAL];
    float ttos=x[S_TYPE_TOS], tsos=x[S_TYPE_SOS], tr2=x[S_TYPE_R2], tr3=x[S_TYPE_R3];
    float alive = (1.0f - sigmoidf(SCALE*(x[S_HALT]-0.5f)))
               * (1.0f - sigmoidf(SCALE*(x[S_AD_IS_BACKWARD]-0.5f))); /* suppress during backward */

    /* Universal: clear output and HAS_OUT */
    out[S_OUTPUT] = -1.0f - x[S_OUTPUT];
    out[S_HAS_OUT] = -x[S_HAS_OUT];
    /* Universal: clear intermediate dims
     * Zone A transient (16-31), Zone D transient (112-127), and arena op
     * transients are cleared here.
     * EXCEPT S_AD_IS_BACKWARD (113) — must persist through Layers 4 and 5
     * which need it for parent load and gradient write-back gating. */
    for (int i = S_OPCODE; i <= S_ABS_DELTA; i++) out[i] += -x[i];
    out[S_AD_IS_FORWARD] += -x[S_AD_IS_FORWARD]; /* clear 112 */
    /* S_AD_IS_BACKWARD (113) intentionally NOT cleared */
    for (int i = S_AD_GRAD_ACCUM; i <= S_AD_SPARE8; i++) out[i] += -x[i];
    for (int i = S_ARENA_TRANSIENT_START; i <= S_ARENA_TRANSIENT_END; i++) out[i] += -x[i];

    float g;
    /* OP_NOP (0) */  g=indicator(op,0)*alive; out[S_PC]+=g;
    /* OP_CONST (1) */g=indicator(op,1)*alive; out[S_PC]+=g; out[S_TOS]+=g*(oper-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(ttos-tsos); out[S_TYPE_R2]+=g*(tsos-tr2); out[S_TYPE_R3]+=g*(tr2-tr3);
    /* OP_NIL (2) */  g=indicator(op,2)*alive; out[S_PC]+=g; out[S_TOS]+=g*(-1-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    out[S_TYPE_TOS]+=g*(TYPE_NIL-ttos); out[S_TYPE_SOS]+=g*(ttos-tsos); out[S_TYPE_R2]+=g*(tsos-tr2); out[S_TYPE_R3]+=g*(tr2-tr3);
    /* OP_TRUE (3) */ g=indicator(op,3)*alive; out[S_PC]+=g; out[S_TOS]+=g*(1-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos); out[S_TYPE_SOS]+=g*(ttos-tsos); out[S_TYPE_R2]+=g*(tsos-tr2); out[S_TYPE_R3]+=g*(tr2-tr3);
    /* OP_FALSE (4) */g=indicator(op,4)*alive; out[S_PC]+=g; out[S_TOS]+=g*(0-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos); out[S_TYPE_SOS]+=g*(ttos-tsos); out[S_TYPE_R2]+=g*(tsos-tr2); out[S_TYPE_R3]+=g*(tr2-tr3);
    /* OP_POP (5) */  g=indicator(op,5)*alive; out[S_PC]+=g; out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(tsos-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_DUP (6) */  g=indicator(op,6)*alive; out[S_PC]+=g; out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    out[S_TYPE_SOS]+=g*(ttos-tsos); out[S_TYPE_R2]+=g*(tsos-tr2); out[S_TYPE_R3]+=g*(tr2-tr3);
    /* OP_SWAP (83): exchange TOS<->SOS (value + type); depth, R2, R3 unchanged */
    g=indicator(op,83)*alive; out[S_PC]+=g; out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(tos-sos);
    out[S_TYPE_TOS]+=g*(tsos-ttos); out[S_TYPE_SOS]+=g*(ttos-tsos);
    /* OP_ADD (7) */  g=indicator(op,7)*alive; out[S_PC]+=g; out[S_TOS]+=g*sos; out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_SUB (8) */  g=indicator(op,8)*alive; out[S_PC]+=g; out[S_TOS]+=g*(sos-2*tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_MUL (9) */  g=indicator(op,9)*alive; out[S_PC]+=g; out[S_TOS]+=g*(product-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_NEG (12) */ g=indicator(op,12)*alive; out[S_PC]+=g; out[S_TOS]+=g*(-2*tos); out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos);
    /* OP_ABS (13) */ g=indicator(op,13)*alive; out[S_PC]+=g; out[S_TOS]+=g*x[S_ABS_DELTA]; out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos);
    /* OP_EQ (14) */  g=indicator(op,14)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_CMP_EQ]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_LT (15): SOS < TOS → CMP_LT precomputed as sigmoid(SCALE*(TOS-SOS-0.5)) */
    g=indicator(op,15)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_CMP_LT]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_GT (16): SOS > TOS → 1 - CMP_LT - CMP_EQ */
    g=indicator(op,16)*alive; out[S_PC]+=g; out[S_TOS]+=g*(1.0f-x[S_CMP_LT]-x[S_CMP_EQ]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_LE (17): SOS <= TOS → CMP_LT + CMP_EQ */
    g=indicator(op,17)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_CMP_LT]+x[S_CMP_EQ]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_GE (18): SOS >= TOS → 1 - CMP_LT */
    g=indicator(op,18)*alive; out[S_PC]+=g; out[S_TOS]+=g*(1.0f-x[S_CMP_LT]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_DIV/OP_MOD: Layer 2 uses S_IS_NATIVE as a transient bounded-active
     * flag; Layer 3 consumes and clears it, so the traced native flag stays
     * false for encoded operands. */
    g=x[S_IS_NATIVE]*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_ZPC1]-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    out[S_HALT]+=indicator(op,10)*alive*indicator(tos,0.0f);
    out[S_HALT]+=indicator(op,11)*alive*indicator(tos,0.0f);
    /* OP_NOT (19): uses ZOPER trick — encode NOT with operand=1 so ZOPER = indicator(TOS,0)*1 */
    g=indicator(op,19)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_ZOPER]-tos); out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos);
    /* OP_GET_LOCAL (20): push mem[operand] — LOADVAL precomputed from OPERAND in layer 2 */
    g=indicator(op,20)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(lv-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(ttos-tsos); out[S_TYPE_R2]+=g*(tsos-tr2); out[S_TYPE_R3]+=g*(tr2-tr3);
    /* OP_SET_LOCAL (21): mem[operand]=TOS, pop — uses precomputed STORED deltas from layer 2 */
    g=indicator(op,21)*alive; {
        out[S_PC]+=g;
        for (int a = 0; a < MEM_SIZE; a++) out[S_MEM0+a] += g * x[S_STORED0+a];
        out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
        out[S_TYPE_TOS]+=g*(tsos-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    }
    /* OP_CONS (31): allocate pair in bounded arena. TOS=cdr, SOS=car. */
    g=indicator(op,31)*alive; {
        float cell = x[S_ARENA_NEXT];
        out[S_PC]+=g;
        out[S_TOS]+=g*(cell-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
        out[S_TYPE_TOS]+=g*(TYPE_PAIR-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
        out[S_ARENA_WRITE_KIND]+=g; out[S_ARENA_WRITE_CAR]+=g; out[S_ARENA_WRITE_CDR]+=g;
        out[S_ARENA_TARGET]+=g*cell;
        out[S_ARENA_NEW_KIND]+=g*ARENA_KIND_PAIR;
        out[S_ARENA_NEW_CAR]+=g*sos; out[S_ARENA_NEW_CDR]+=g*tos;
        out[S_ARENA_NEW_CAR_TYPE]+=g*tsos; out[S_ARENA_NEW_CDR_TYPE]+=g*ttos;
        out[S_ARENA_NEXT]+=g;
    }
    /* OP_CAR (32): default to 0, then layer4 overwrites from arena[target].car. */
    g=indicator(op,32)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(-tos); out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos);
    out[S_ARENA_READ_CAR]+=g; out[S_ARENA_TARGET]+=g*tos;
    /* OP_CDR (33): default to 0, then layer4 overwrites from arena[target].cdr. */
    g=indicator(op,33)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(-tos); out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos);
    out[S_ARENA_READ_CDR]+=g; out[S_ARENA_TARGET]+=g*tos;
    /* OP_NULL_P (34): weight-encoded nil predicate */
    g=indicator(op,34)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_TYPE_IS_NIL]-tos); out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos);
    /* OP_GET_UPVALUE (22): MEM fallback first; Layer 4 overwrites TOS from
     * current arena closure cell when S_CUR_CLOSURE points at one. */
    g=indicator(op,22)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(lv-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(ttos-tsos); out[S_TYPE_R2]+=g*(tsos-tr2); out[S_TYPE_R3]+=g*(tr2-tr3);
    out[S_ARENA_READ_CAR]+=g; out[S_ARENA_TARGET]+=g*(x[S_CUR_CLOSURE]+oper+1.0f);
    /* OP_SET_UPVALUE (23): write MEM fallback and current arena closure cell, then pop. */
    g=indicator(op,23)*alive; {
        out[S_PC]+=g;
        for (int a = 0; a < MEM_SIZE; a++) out[S_MEM0+a] += g * x[S_STORED0+a];
        out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
        out[S_TYPE_TOS]+=g*(tsos-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
        out[S_ARENA_WRITE_CAR]+=g; out[S_ARENA_TARGET]+=g*(x[S_CUR_CLOSURE]+oper+1.0f);
        out[S_ARENA_NEW_CAR]+=g*tos; out[S_ARENA_NEW_CAR_TYPE]+=g*ttos;
    }
    /* OP_CLOSE_UPVALUE (38): close MEM[operand] into the current arena closure cell. */
    g=indicator(op,38)*alive;
    out[S_PC]+=g; out[S_ARENA_WRITE_CAR]+=g;
    out[S_ARENA_TARGET]+=g*(x[S_CUR_CLOSURE]+oper+1.0f);
    out[S_ARENA_NEW_CAR]+=g*lv; out[S_ARENA_NEW_CAR_TYPE]+=g*TYPE_NUMBER;
    /* OP_OPEN_CLOSURE (54): make TOS the current arena closure without
     * disturbing the operand stack. */
    g=indicator(op,54)*alive;
    out[S_PC]+=g; out[S_CUR_CLOSURE]+=g*(tos-x[S_CUR_CLOSURE]);
    /* OP_CLOSURE (24): allocate closure header plus four bounded upvalue
     * cells in the arena. car stores function entry PC; cdr stores capacity. */
    g=indicator(op,24)*alive; {
        float cell = x[S_ARENA_NEXT];
        out[S_PC]+=g;
        out[S_TOS]+=g*(cell-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
        out[S_TYPE_TOS]+=g*(TYPE_CLOSURE-ttos); out[S_TYPE_SOS]+=g*(ttos-tsos); out[S_TYPE_R2]+=g*(tsos-tr2); out[S_TYPE_R3]+=g*(tr2-tr3);
        out[S_ARENA_WRITE_KIND]+=g; out[S_ARENA_WRITE_CAR]+=g; out[S_ARENA_WRITE_CDR]+=g;
        out[S_ARENA_TARGET]+=g*cell;
        out[S_ARENA_NEW_KIND]+=g*ARENA_KIND_CLOSURE;
        out[S_ARENA_NEW_CAR]+=g*oper; out[S_ARENA_NEW_CDR]+=g*(float)MEM_SIZE;
        out[S_ARENA_NEW_CAR_TYPE]+=g*TYPE_NUMBER; out[S_ARENA_NEW_CDR_TYPE]+=g*TYPE_NUMBER;
        out[S_ARENA_NEXT]+=g*(float)(1 + MEM_SIZE);
    }
    /* OP_TAIL_CALL (26): frame reuse. Encode the stack/register shuffle
     * directly; CALL frame handling remains native only for non-tail calls. */
    g=indicator(op,26)*alive; {
        float argc0 = indicator(oper, 0.0f);
        float argc1 = indicator(oper, 1.0f);
        float argc2 = indicator(oper, 2.0f);
        float argc3 = indicator(oper, 3.0f);
        float argc4 = indicator(oper, 4.0f);
        float argc = argc1 + 2.0f*argc2 + 3.0f*argc3 + 4.0f*argc4;
        (void)argc0;
        out[S_PC]+=g*(tos-x[S_PC]);
        out[S_MEM0]+=g*((argc1+argc2+argc3+argc4)*sos - x[S_MEM0]);
        out[S_MEM1]+=g*((argc2+argc3+argc4)*r2 - x[S_MEM1]);
        out[S_MEM2]+=g*((argc3+argc4)*r3 - x[S_MEM2]);
        out[S_MEM3]+=g*(-x[S_MEM3]);
        out[S_TOS]+=g*(-tos); out[S_SOS]+=g*(-sos); out[S_R2]+=g*(-r2); out[S_R3]+=g*(-r3);
        out[S_DEPTH]+=g*(-1.0f-argc);
        out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(TYPE_NUMBER-tsos);
        out[S_TYPE_R2]+=g*(TYPE_NUMBER-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    }
    /* OP_NATIVE_CALL (37): delegate */
    g=indicator(op,37)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;

    /* OP_CALLCC (55): bounded escape continuation. Capture the modeled VM
     * registers and MEM bank into four contiguous arena cells, then jump to
     * the direct-entry function PC in TOS with the continuation in MEM0. */
    g=indicator(op,55)*alive; {
        int elem_dims[4] = { S_ARENA_LIST_E0, S_ARENA_LIST_E1, S_ARENA_LIST_E2, S_ARENA_LIST_E3 };
        int elem_type_dims[4] = { S_ARENA_LIST_T0, S_ARENA_LIST_T1, S_ARENA_LIST_T2, S_ARENA_LIST_T3 };
        int cdr_dims[4] = { S_ARENA_LIST_CDR0, S_ARENA_LIST_CDR1, S_ARENA_LIST_CDR2, S_ARENA_LIST_CDR3 };
        int cdr_type_dims[4] = { S_ARENA_LIST_CDRT0, S_ARENA_LIST_CDRT1, S_ARENA_LIST_CDRT2, S_ARENA_LIST_CDRT3 };
        int has_dims[4] = { S_ARENA_LIST_HAS_E0, S_ARENA_LIST_HAS_E1, S_ARENA_LIST_HAS_E2, S_ARENA_LIST_HAS_E3 };
        float base = x[S_ARENA_NEXT];

        out[S_PC]+=g*(tos-x[S_PC]);
        out[S_MEM0]+=g*(base-x[S_MEM0]); out[S_MEM1]+=g*(-x[S_MEM1]);
        out[S_MEM2]+=g*(-x[S_MEM2]); out[S_MEM3]+=g*(-x[S_MEM3]);
        out[S_TOS]+=g*(-tos); out[S_SOS]+=g*(-sos); out[S_R2]+=g*(-r2); out[S_R3]+=g*(-r3);
        out[S_DEPTH]+=g*(-x[S_DEPTH]);
        out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(TYPE_NUMBER-tsos);
        out[S_TYPE_R2]+=g*(TYPE_NUMBER-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
        out[S_ARENA_LIST_BASE]+=g*base;
        out[S_ARENA_NEXT]+=g*(float)ARENA_CONT_CELLS;

        out[elem_dims[0]] += g*(x[S_PC] + 1.0f);
        out[cdr_dims[0]] += g*(x[S_DEPTH] - 1.0f);
        out[elem_type_dims[0]] += g*sos;
        out[cdr_type_dims[0]] += g*r2;
        out[elem_dims[1]] += g*r3;
        out[cdr_dims[1]] += 0.0f;
        out[elem_type_dims[1]] += g*tsos;
        out[cdr_type_dims[1]] += g*tr2;
        out[elem_dims[2]] += g*tr3;
        out[cdr_dims[2]] += g*TYPE_NUMBER;
        out[elem_type_dims[2]] += g*x[S_MEM0];
        out[cdr_type_dims[2]] += g*x[S_MEM1];
        out[elem_dims[3]] += g*x[S_MEM2];
        out[cdr_dims[3]] += g*x[S_MEM3];
        out[elem_type_dims[3]] += g*x[S_WIND_DEPTH];
        out[cdr_type_dims[3]] += 0.0f;
        for (int i = 0; i < ARENA_CONT_CELLS; i++) out[has_dims[i]] += g;
    }

    /* OP_INVOKE_CC (56): Layer 3 marks a continuation restore; Layer 4 reads
     * the arena record and overwrites PC/MEM/stack. TOS remains the return
     * value, and SOS supplies the continuation base cell. */
    g=indicator(op,56)*alive;
    out[S_ARENA_VEC_HAS_E0]+=g;
    out[S_ARENA_VEC_BASE]+=g*sos;
    out[S_ARENA_VEC_LEN]+=g*CONT_RESTORE_MARKER;

    /* Stage-1 type predicates (45-50): weight-encoded via Layer 1 type indicators. */
    g=indicator(op,45)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_TYPE_IS_PAIR]-tos); out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos);
    g=indicator(op,46)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_TYPE_IS_NUM]-tos);  out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos);
    g=indicator(op,47)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_TYPE_IS_STR]-tos);  out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos);
    g=indicator(op,48)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_TYPE_IS_BOOL]-tos); out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos);
    g=indicator(op,49)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_TYPE_IS_PROC]-tos); out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos);
    g=indicator(op,50)*alive; out[S_PC]+=g; out[S_TOS]+=g*(x[S_TYPE_IS_VEC]-tos);  out[S_TYPE_TOS]+=g*(TYPE_BOOL-ttos);

    /* OP_SET_CAR (51): mutate arena pair car, then pop pair+value. */
    g=indicator(op,51)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(r2-tos); out[S_SOS]+=g*(r3-sos); out[S_R2]+=g*(-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-2);
    out[S_TYPE_TOS]+=g*(tr2-ttos); out[S_TYPE_SOS]+=g*(tr3-tsos); out[S_TYPE_R2]+=g*(TYPE_NUMBER-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    out[S_ARENA_WRITE_CAR]+=g; out[S_ARENA_TARGET]+=g*sos;
    out[S_ARENA_NEW_CAR]+=g*tos; out[S_ARENA_NEW_CAR_TYPE]+=g*ttos;
    /* OP_SET_CDR (52): mutate arena pair cdr, then pop pair+value. */
    g=indicator(op,52)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(r2-tos); out[S_SOS]+=g*(r3-sos); out[S_R2]+=g*(-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-2);
    out[S_TYPE_TOS]+=g*(tr2-ttos); out[S_TYPE_SOS]+=g*(tr3-tsos); out[S_TYPE_R2]+=g*(TYPE_NUMBER-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    out[S_ARENA_WRITE_CDR]+=g; out[S_ARENA_TARGET]+=g*sos;
    out[S_ARENA_NEW_CDR]+=g*tos; out[S_ARENA_NEW_CDR_TYPE]+=g*ttos;

    /* OP_VEC_CREATE (39): bounded inline vector. Header cell at ARENA_NEXT,
     * followed by up to four contiguous element cells. */
    g=indicator(op,39)*alive; out[S_PC]+=g;
    {
        int elem_dims[4] = { S_ARENA_VEC_E0, S_ARENA_VEC_E1, S_ARENA_VEC_E2, S_ARENA_VEC_E3 };
        int elem_type_dims[4] = { S_ARENA_VEC_T0, S_ARENA_VEC_T1, S_ARENA_VEC_T2, S_ARENA_VEC_T3 };
        int elem_has_dims[4] = { S_ARENA_VEC_HAS_E0, S_ARENA_VEC_HAS_E1, S_ARENA_VEC_HAS_E2, S_ARENA_VEC_HAS_E3 };
        float vals[4] = { tos, sos, r2, r3 };
        float types[4] = { ttos, tsos, tr2, tr3 };
        for (int count = 0; count <= ARENA_MAX_INLINE_VECTOR; count++) {
            float gc = g * indicator(oper, (float)count);
            out[S_TOS]+=gc*(x[S_ARENA_NEXT]-tos); out[S_SOS]+=gc*(-sos); out[S_R2]+=gc*(-r2); out[S_R3]+=gc*(-r3);
            out[S_DEPTH]+=gc*(1.0f-(float)count);
            out[S_TYPE_TOS]+=gc*(TYPE_VECTOR-ttos); out[S_TYPE_SOS]+=gc*(TYPE_NUMBER-tsos);
            out[S_TYPE_R2]+=gc*(TYPE_NUMBER-tr2); out[S_TYPE_R3]+=gc*(TYPE_NUMBER-tr3);
            out[S_ARENA_VEC_WRITE]+=gc;
            out[S_ARENA_VEC_BASE]+=gc*x[S_ARENA_NEXT];
            out[S_ARENA_VEC_LEN]+=gc*(float)count;
            for (int i = 0; i < count; i++) {
                int src = count - 1 - i;
                out[elem_dims[i]] += gc * vals[src];
                out[elem_type_dims[i]] += gc * types[src];
                out[elem_has_dims[i]] += gc;
            }
            out[S_ARENA_NEXT]+=gc*(float)(count + 1);
        }
    }

    /* OP_VEC_REF (40): TOS=index, SOS=vector header. Element cells are
     * contiguous, so element target = header + 1 + index. Layer 4 reads car. */
    g=indicator(op,40)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    out[S_ARENA_READ_CAR]+=g; out[S_ARENA_TARGET]+=g*(sos + tos + 1.0f);

    /* OP_VEC_SET (41): TOS=value, SOS=index, R2=vector header. */
    g=indicator(op,41)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(r3-tos); out[S_SOS]+=g*(-sos); out[S_R2]+=g*(-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-3);
    out[S_TYPE_TOS]+=g*(tr3-ttos); out[S_TYPE_SOS]+=g*(TYPE_NUMBER-tsos); out[S_TYPE_R2]+=g*(TYPE_NUMBER-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    out[S_ARENA_WRITE_CAR]+=g; out[S_ARENA_TARGET]+=g*(r2 + sos + 1.0f);
    out[S_ARENA_NEW_CAR]+=g*tos; out[S_ARENA_NEW_CAR_TYPE]+=g*ttos;

    /* OP_VEC_LEN (42): vector header's car field stores length. */
    g=indicator(op,42)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(-tos); out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos);
    out[S_ARENA_READ_CAR]+=g; out[S_ARENA_TARGET]+=g*tos;

    /* OP_STR_REF (43) and OP_STR_LEN (44): strings use the same arena
     * length-header + contiguous element layout as bounded vectors. */
    g=indicator(op,43)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    out[S_ARENA_READ_CAR]+=g; out[S_ARENA_TARGET]+=g*(sos + tos + 1.0f);

    g=indicator(op,44)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(-tos); out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos);
    out[S_ARENA_READ_CAR]+=g; out[S_ARENA_TARGET]+=g*tos;

    /* OP_POPN (53): weight-encoded for current compiler emissions n <= 3.
     * It removes N values below TOS while keeping TOS itself. */
    g=indicator(op,53)*alive; out[S_PC]+=g;
    float gp1=g*indicator(oper,1.0f);
    out[S_SOS]+=gp1*(r2-sos); out[S_R2]+=gp1*(r3-r2); out[S_R3]+=gp1*(-r3); out[S_DEPTH]+=gp1*(-1);
    out[S_TYPE_SOS]+=gp1*(tr2-tsos); out[S_TYPE_R2]+=gp1*(tr3-tr2); out[S_TYPE_R3]+=gp1*(TYPE_NUMBER-tr3);
    float gp2=g*indicator(oper,2.0f);
    out[S_SOS]+=gp2*(r3-sos); out[S_R2]+=gp2*(-r2); out[S_R3]+=gp2*(-r3); out[S_DEPTH]+=gp2*(-2);
    out[S_TYPE_SOS]+=gp2*(tr3-tsos); out[S_TYPE_R2]+=gp2*(TYPE_NUMBER-tr2); out[S_TYPE_R3]+=gp2*(TYPE_NUMBER-tr3);
    float gp3=g*indicator(oper,3.0f);
    out[S_SOS]+=gp3*(-sos); out[S_R2]+=gp3*(-r2); out[S_R3]+=gp3*(-r3); out[S_DEPTH]+=gp3*(-3);
    out[S_TYPE_SOS]+=gp3*(TYPE_NUMBER-tsos); out[S_TYPE_R2]+=gp3*(TYPE_NUMBER-tr2); out[S_TYPE_R3]+=gp3*(TYPE_NUMBER-tr3);

    /* OP_VOID (63): no stack effect, PC++ */
    g=indicator(op,63)*alive; out[S_PC]+=g;

    /* Exception/dynamic-wind bookkeeping. These keep bounded counters and
     * stack effects in the residual stream; full raise/unwind remains native. */
    g=indicator(op,57)*alive; out[S_PC]+=g; out[S_EXC_DEPTH]+=g;
    g=indicator(op,58)*alive; out[S_PC]+=g; out[S_EXC_DEPTH]+=g*(-1);
    g=indicator(op,59)*alive;
    out[S_PC]+=g; out[S_TOS]+=g*(-tos); out[S_SOS]+=g*(tos-sos); out[S_R2]+=g*(sos-r2); out[S_R3]+=g*(r2-r3); out[S_DEPTH]+=g;
    out[S_TYPE_TOS]+=g*(TYPE_NUMBER-ttos); out[S_TYPE_SOS]+=g*(ttos-tsos); out[S_TYPE_R2]+=g*(tsos-tr2); out[S_TYPE_R3]+=g*(tr2-tr3);
    g=indicator(op,61)*alive;
    out[S_PC]+=g; out[S_WIND_DEPTH]+=g; out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(tsos-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    g=indicator(op,62)*alive; out[S_PC]+=g; out[S_WIND_DEPTH]+=g*(-1);

    /* OP_PACK_REST (60): pack MEM[n_fixed..3] into an arena list and
     * store the resulting list pointer back in MEM[n_fixed]. */
    g=indicator(op,60)*alive; out[S_PC]+=g; {
        int elem_dims[4] = { S_ARENA_LIST_E0, S_ARENA_LIST_E1, S_ARENA_LIST_E2, S_ARENA_LIST_E3 };
        int elem_type_dims[4] = { S_ARENA_LIST_T0, S_ARENA_LIST_T1, S_ARENA_LIST_T2, S_ARENA_LIST_T3 };
        int cdr_dims[4] = { S_ARENA_LIST_CDR0, S_ARENA_LIST_CDR1, S_ARENA_LIST_CDR2, S_ARENA_LIST_CDR3 };
        int cdr_type_dims[4] = { S_ARENA_LIST_CDRT0, S_ARENA_LIST_CDRT1, S_ARENA_LIST_CDRT2, S_ARENA_LIST_CDRT3 };
        int has_dims[4] = { S_ARENA_LIST_HAS_E0, S_ARENA_LIST_HAS_E1, S_ARENA_LIST_HAS_E2, S_ARENA_LIST_HAS_E3 };
        for (int n_fixed = 0; n_fixed <= MEM_SIZE; n_fixed++) {
            float gf = g * indicator(oper, (float)n_fixed);
            int count = MEM_SIZE - n_fixed;
            out[S_ARENA_LIST_BASE] += gf * x[S_ARENA_NEXT];
            out[S_ARENA_NEXT] += gf * (float)count;
            if (n_fixed < MEM_SIZE)
                out[S_MEM0+n_fixed] += gf * ((count > 0 ? x[S_ARENA_NEXT] : -1.0f) - x[S_MEM0+n_fixed]);
            for (int j = 0; j < count; j++) {
                int mem_dim = S_MEM0 + n_fixed + j;
                float cdr = (j + 1 < count) ? x[S_ARENA_NEXT] + (float)(j + 1) : -1.0f;
                float cdr_type = (j + 1 < count) ? TYPE_PAIR : TYPE_NIL;
                out[elem_dims[j]] += gf * x[mem_dim];
                out[elem_type_dims[j]] += gf * TYPE_NUMBER;
                out[cdr_dims[j]] += gf * cdr;
                out[cdr_type_dims[j]] += gf * cdr_type;
                out[has_dims[j]] += gf;
            }
        }
    }

    /* Remaining delegated opcodes (38-62): all set IS_NATIVE + PC++ */
    for (int opc = 38; opc <= 62; opc++) {
        if (opc == 38 || (opc >= 39 && opc <= 44) || (opc >= 45 && opc <= 50) ||
            opc == 51 || opc == 52 || opc == 53 || opc == 54 ||
            opc == 55 || opc == 56 || (opc >= 57 && opc <= 62)) continue;
        g=indicator(op,(float)opc)*alive; out[S_PC]+=g; out[S_IS_NATIVE]+=g;
    }
    /* OP_CALL (25): set IS_CALL flag for exec loop, PC++ */
    g=indicator(op,25)*alive; out[S_PC]+=g; out[S_IS_CALL]+=g;
    /* OP_RETURN (27): set IS_RET flag for exec loop, PC++ */
    g=indicator(op,27)*alive; out[S_PC]+=g; out[S_IS_RET]+=g;
    /* OP_JUMP (28) */
    g=indicator(op,28)*alive; out[S_PC]+=g*(oper-x[S_PC]);
    /* OP_JUMP_IF_FALSE (29): if TOS==0 goto operand, else PC+1. Pop TOS.
     * delta[PC] = 1 + ZOPER - ZPC1 (inverted from v1's JUMP_IF) */
    g=indicator(op,29)*alive;
    out[S_PC]+=g*(1.0f + x[S_ZOPER] - x[S_ZPC1]);
    out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(tsos-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_LOOP (30): unconditional backward jump */
    g=indicator(op,30)*alive; out[S_PC]+=g*(oper-x[S_PC]);
    /* OP_PRINT (35): output = TOS, pop */
    g=indicator(op,35)*alive; out[S_PC]+=g; out[S_OUTPUT]+=g*(tos+1); out[S_HAS_OUT]+=g;
    out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3); out[S_DEPTH]+=g*(-1);
    out[S_TYPE_TOS]+=g*(tsos-ttos); out[S_TYPE_SOS]+=g*(tr2-tsos); out[S_TYPE_R2]+=g*(tr3-tr2); out[S_TYPE_R3]+=g*(TYPE_NUMBER-tr3);
    /* OP_HALT (36) */
    g=indicator(op,36)*alive; out[S_HALT]+=g;

    /* ── AD Forward Ops (64-78) ──
     * These record nodes on the embedded tape. The actual tape WRITE happens
     * in layer4_ffn — layer3 computes the node fields and sets AD_IS_FORWARD. */
    float tlen = x[S_AD_TAPE_LEN];
    float not_backward = 1.0f - sigmoidf(SCALE * (x[S_AD_IS_BACKWARD] - 0.5f)); /* 1 when NOT in backward */
    float tape_ok = sigmoidf(SCALE * ((float)AD_MAX_TAPE - 0.5f - tlen)) * not_backward;

    /* OP_AD_VAR (64): record variable node, push tape index */
    g=indicator(op,64)*alive*tape_ok;
    out[S_AD_CUR_OP]    += g * AD_OP_VAR;
    out[S_AD_CUR_VALUE]  += g * oper;
    out[S_AD_CUR_LEFT]   += g * (-1);
    out[S_AD_CUR_RIGHT]  += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_R3]+=g*(r2-r3); out[S_R2]+=g*(sos-r2); out[S_SOS]+=g*(tos-sos);
    out[S_TOS]+=g*(tlen-tos); /* push tape index */
    out[S_DEPTH]+=g; out[S_PC]+=g; out[S_AD_MODE]+=g*(1.0f-x[S_AD_MODE]);

    /* OP_AD_CONST (65): record constant node */
    g=indicator(op,65)*alive*tape_ok;
    out[S_AD_CUR_OP]    += g * AD_OP_CONST;
    out[S_AD_CUR_VALUE]  += g * oper;
    out[S_AD_CUR_LEFT]   += g * (-1);
    out[S_AD_CUR_RIGHT]  += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_R3]+=g*(r2-r3); out[S_R2]+=g*(sos-r2); out[S_SOS]+=g*(tos-sos);
    out[S_TOS]+=g*(tlen-tos);
    out[S_DEPTH]+=g; out[S_PC]+=g; out[S_AD_MODE]+=g*(1.0f-x[S_AD_MODE]);

    /* Binary AD ops (66-68): AD_ADD, AD_SUB, AD_MUL
     * TOS=right_idx, SOS=left_idx. Pop both, push tape index.
     * Value computation: read parent values from tape via layer2 loaded fields.
     * BUT: layer2 loaded AD_CUR_* based on AD_CURSOR, not on TOS/SOS.
     * For forward ops, we need parent values at TOS and SOS indices.
     * Since we can't do two random-access loads in one pass, we use the
     * OPERAND trick: layer2 loads based on cursor, but for forward ops the
     * values come from register stack. The reference interpreter reads them
     * directly. For the simulated path, we compute values from the tape. */

    /* Helper: read tape value at index i (simulated via indicator sum) */
    /* For AD_ADD: left=SOS, right=TOS. We need tape[SOS].value and tape[TOS].value */
    float ad_left_val = 0, ad_right_val = 0;
    for (int i = 0; i < AD_MAX_TAPE; i++) {
        float li = indicator(sos, (float)i);
        float ri = indicator(tos, (float)i);
        ad_left_val  += li * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
        ad_right_val += ri * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
    }

    /* OP_AD_ADD (66) */
    g=indicator(op,66)*alive*tape_ok;
    out[S_AD_CUR_OP]    += g * AD_OP_ADD;
    out[S_AD_CUR_VALUE]  += g * (ad_left_val + ad_right_val);
    out[S_AD_CUR_LEFT]   += g * sos;
    out[S_AD_CUR_RIGHT]  += g * tos;
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3);
    out[S_DEPTH]+=g*(-1); out[S_PC]+=g;

    /* OP_AD_SUB (67) */
    g=indicator(op,67)*alive*tape_ok;
    out[S_AD_CUR_OP]    += g * AD_OP_SUB;
    out[S_AD_CUR_VALUE]  += g * (ad_left_val - ad_right_val);
    out[S_AD_CUR_LEFT]   += g * sos;
    out[S_AD_CUR_RIGHT]  += g * tos;
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3);
    out[S_DEPTH]+=g*(-1); out[S_PC]+=g;

    /* OP_AD_MUL (68) */
    g=indicator(op,68)*alive*tape_ok;
    out[S_AD_CUR_OP]    += g * AD_OP_MUL;
    out[S_AD_CUR_VALUE]  += g * (ad_left_val * ad_right_val);
    out[S_AD_CUR_LEFT]   += g * sos;
    out[S_AD_CUR_RIGHT]  += g * tos;
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3);
    out[S_DEPTH]+=g*(-1); out[S_PC]+=g;

    /* OP_AD_DIV (79) */
    g=indicator(op,79)*alive*tape_ok; {
    float safe_div_right = fabsf(ad_right_val) > 1e-15f ? ad_right_val :
                           (ad_right_val < 0 ? -1e-15f : 1e-15f);
    out[S_AD_CUR_OP]    += g * AD_OP_DIV;
    out[S_AD_CUR_VALUE]  += g * (ad_left_val / safe_div_right);
    out[S_AD_CUR_SAVED]  += g * (1.0f / safe_div_right);
    out[S_AD_CUR_LEFT]   += g * sos;
    out[S_AD_CUR_RIGHT]  += g * tos;
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3);
    out[S_DEPTH]+=g*(-1); out[S_PC]+=g; }

    /* OP_AD_POW (80) */
    g=indicator(op,80)*alive*tape_ok; {
    if (g > 0.5f) {
        float safe_pow_left = ad_left_val > 1e-15f ? ad_left_val : 1e-15f;
        out[S_AD_CUR_OP]    += AD_OP_POW;
        out[S_AD_CUR_VALUE]  += powf(safe_pow_left, ad_right_val);
        out[S_AD_CUR_SAVED]  += ad_right_val * powf(safe_pow_left, ad_right_val - 1.0f);
        out[S_AD_CUR_LEFT]   += sos;
        out[S_AD_CUR_RIGHT]  += tos;
        out[S_AD_IS_FORWARD] += 1.0f;
        out[S_TOS]+=(tlen-tos); out[S_SOS]+=(r2-sos); out[S_R2]+=(r3-r2); out[S_R3]+=(-r3);
        out[S_DEPTH]+=-1.0f; out[S_PC]+=1.0f;
    } }

    /* Unary AD ops (69-76, 81-82): AD_NEG, AD_ABS, AD_RELU, AD_SIGMOID, AD_TANH,
     * AD_EXP, AD_LOG, AD_SQRT, AD_SIN, AD_COS.
     * TOS=input_idx. Replace TOS with tape index.
     * Input value from tape at TOS index: */
    float ad_input_val = 0;
    for (int i = 0; i < AD_MAX_TAPE; i++)
        ad_input_val += indicator(tos, (float)i) * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];

    /* OP_AD_NEG (69) */
    g=indicator(op,69)*alive*tape_ok;
    out[S_AD_CUR_OP] += g * AD_OP_NEG;
    out[S_AD_CUR_VALUE] += g * (-ad_input_val);
    out[S_AD_CUR_SAVED] += g * (-1.0f); /* derivative factor: d(-x)/dx = -1 */
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g;

    /* OP_AD_ABS (70) */
    g=indicator(op,70)*alive*tape_ok; {
    float v70 = fabsf(ad_input_val);
    out[S_AD_CUR_OP] += g * AD_OP_ABS;
    out[S_AD_CUR_VALUE] += g * v70;
    out[S_AD_CUR_SAVED] += g * ((ad_input_val > 0) ? 1.0f : (ad_input_val < 0) ? -1.0f : 0.0f);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g; }

    /* OP_AD_SIN (81) */
    g=indicator(op,81)*alive*tape_ok; {
    out[S_AD_CUR_OP] += g * AD_OP_SIN;
    out[S_AD_CUR_VALUE] += g * sinf(ad_input_val);
    out[S_AD_CUR_SAVED] += g * cosf(ad_input_val);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g; }

    /* OP_AD_COS (82) */
    g=indicator(op,82)*alive*tape_ok; {
    out[S_AD_CUR_OP] += g * AD_OP_COS;
    out[S_AD_CUR_VALUE] += g * cosf(ad_input_val);
    out[S_AD_CUR_SAVED] += g * -sinf(ad_input_val);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g; }

    /* OP_AD_RELU (71) */
    g=indicator(op,71)*alive*tape_ok; {
    out[S_AD_CUR_OP] += g * AD_OP_RELU;
    out[S_AD_CUR_VALUE] += g * (ad_input_val > 0 ? ad_input_val : 0);
    out[S_AD_CUR_SAVED] += g * (ad_input_val > 0 ? 1.0f : 0.0f);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g; }

    /* OP_AD_SIGMOID (72) */
    g=indicator(op,72)*alive*tape_ok; {
    float v72 = 1.0f/(1.0f+expf(-ad_input_val));
    out[S_AD_CUR_OP] += g * AD_OP_SIGMOID;
    out[S_AD_CUR_VALUE] += g * v72;
    out[S_AD_CUR_SAVED] += g * v72 * (1.0f - v72);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g; }

    /* OP_AD_TANH (73) */
    g=indicator(op,73)*alive*tape_ok; {
    float v73 = tanhf(ad_input_val);
    out[S_AD_CUR_OP] += g * AD_OP_TANH;
    out[S_AD_CUR_VALUE] += g * v73;
    out[S_AD_CUR_SAVED] += g * (1.0f - v73 * v73);
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g; }

    /* OP_AD_EXP (74) */
    g=indicator(op,74)*alive*tape_ok; {
    float v74 = expf(ad_input_val);
    out[S_AD_CUR_OP] += g * AD_OP_EXP;
    out[S_AD_CUR_VALUE] += g * v74;
    out[S_AD_CUR_SAVED] += g * v74; /* exp'(x) = exp(x) */
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g; }

    /* OP_AD_LOG (75) */
    g=indicator(op,75)*alive*tape_ok; {
    float safe75 = (ad_input_val > 1e-15f) ? ad_input_val : 1e-15f; /* must be positive for logf */
    out[S_AD_CUR_OP] += g * AD_OP_LOG;
    out[S_AD_CUR_VALUE] += g * logf(safe75);
    out[S_AD_CUR_SAVED] += g / safe75; /* log'(x) = 1/x */
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g; }

    /* OP_AD_SQRT (76) */
    g=indicator(op,76)*alive*tape_ok; {
    float v76 = sqrtf(ad_input_val > 0 ? ad_input_val : 0);
    float safe76 = (fabsf(v76) > 1e-15f) ? v76 : 1e-15f;
    out[S_AD_CUR_OP] += g * AD_OP_SQRT;
    out[S_AD_CUR_VALUE] += g * v76;
    out[S_AD_CUR_SAVED] += g / (2.0f * safe76); /* sqrt'(x) = 1/(2*sqrt(x)) */
    out[S_AD_CUR_LEFT] += g * tos;
    out[S_AD_CUR_RIGHT] += g * (-1);
    out[S_AD_IS_FORWARD] += g;
    out[S_TOS]+=g*(tlen-tos); out[S_PC]+=g; }

    /* OP_AD_BACKWARD (77): start backward pass.
     * Seed the output node's gradient = 1.0 directly in the tape.
     * Do NOT set AD_IS_FORWARD (that would trigger a spurious tape write in layer4). */
    g=indicator(op,77)*alive; {
        out[S_AD_MODE] += g * (2.0f - x[S_AD_MODE]);
        out[S_AD_CURSOR] += g * (tos - x[S_AD_CURSOR]);
        out[S_AD_IS_BACKWARD] += g;
        /* Seed gradient: write 1.0 to tape[TOS].gradient via indicator */
        for (int i = 0; i < AD_MAX_TAPE; i++) {
            float ti = indicator(tos, (float)i);
            out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD] += g * ti * 1.0f;
        }
        out[S_TOS]+=g*(sos-tos); out[S_SOS]+=g*(r2-sos); out[S_R2]+=g*(r3-r2); out[S_R3]+=g*(-r3);
        out[S_DEPTH]+=g*(-1); out[S_PC]+=g;
    }

    /* OP_AD_GRAD (78): replace TOS with gradient of tape[TOS] */
    g=indicator(op,78)*alive; {
        /* Read gradient from tape at TOS index */
        float grad_val = 0;
        for (int i = 0; i < AD_MAX_TAPE; i++)
            grad_val += indicator(tos, (float)i) * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD];
        out[S_TOS] += g * (grad_val - tos);
        out[S_PC] += g;
    }

    /* ── AD Backward gradient computation ──
     * When AD_IS_BACKWARD is set (by run_simulated's backward step),
     * compute gradient contributions based on AD_CUR_OP.
     * Results go into AD_LEFT_GRAD_NEW and AD_RIGHT_GRAD_NEW for layer4. */
    float bw = x[S_AD_IS_BACKWARD]; /* > 0 during backward */
    if (bw > 0.5f) {
        float cur_grad = x[S_AD_CUR_GRAD];
        float cur_op = x[S_AD_CUR_OP];
        float cur_val = x[S_AD_CUR_VALUE];
        float gl = x[S_AD_PROD_GRAD_LV]; /* precomputed: grad * right_value (for MUL dL) */
        float gr = x[S_AD_PROD_GRAD_RV]; /* precomputed: grad * left_value (for MUL dR) */

        float gsv = x[S_AD_PROD_GRAD_SV]; /* precomputed: grad * saved_val (from Layer 2 SQUARE) */

        /* Binary ops */
        /* ADD: dL = grad, dR = grad */
        float ba = indicator(cur_op, AD_OP_ADD);
        out[S_AD_LEFT_GRAD_NEW]  += ba * cur_grad;
        out[S_AD_RIGHT_GRAD_NEW] += ba * cur_grad;
        /* SUB: dL = grad, dR = -grad */
        float bs = indicator(cur_op, AD_OP_SUB);
        out[S_AD_LEFT_GRAD_NEW]  += bs * cur_grad;
        out[S_AD_RIGHT_GRAD_NEW] -= bs * cur_grad;
        /* MUL: dL = grad*R, dR = grad*L (precomputed in Layer 2) */
        float bm = indicator(cur_op, AD_OP_MUL);
        out[S_AD_LEFT_GRAD_NEW]  += bm * gl;
        out[S_AD_RIGHT_GRAD_NEW] += bm * gr;
        /* DIV: dL = grad/right via saved reciprocal; dR = -grad*left/(right^2). */
        float bd = indicator(cur_op, AD_OP_DIV);
        float safe_div_right = fabsf(x[S_AD_RIGHT_VALUE]) > 1e-15f ? x[S_AD_RIGHT_VALUE] :
                               (x[S_AD_RIGHT_VALUE] < 0 ? -1e-15f : 1e-15f);
        out[S_AD_LEFT_GRAD_NEW]  += bd * gsv;
        out[S_AD_RIGHT_GRAD_NEW] += bd * gr * (-1.0f / (safe_div_right * safe_div_right));
        /* POW: dL = grad*right*left^(right-1) via saved; dR = grad*value*log(left). */
        float bp = indicator(cur_op, AD_OP_POW);
        float safe_pow_left = x[S_AD_LEFT_VALUE] > 1e-15f ? x[S_AD_LEFT_VALUE] : 1e-15f;
        out[S_AD_LEFT_GRAD_NEW]  += bp * gsv;
        out[S_AD_RIGHT_GRAD_NEW] += bp * cur_grad * cur_val * logf(safe_pow_left);

        /* ALL unary ops: dL = grad * saved_val (precomputed in Layer 2 as AD_PROD_GRAD_SV).
         * saved_val was stored during forward recording:
         *   NEG=-1, ABS=sign, RELU=step, SIGMOID=val*(1-val),
         *   TANH=1-val², EXP=val, LOG=1/input, SQRT=1/(2*val),
         *   SIN=cos(input), COS=-sin(input). */
        float unary_ops[] = { AD_OP_NEG, AD_OP_ABS, AD_OP_RELU, AD_OP_SIGMOID,
                              AD_OP_TANH, AD_OP_EXP, AD_OP_LOG, AD_OP_SQRT,
                              AD_OP_SIN, AD_OP_COS };
        for (int u = 0; u < 10; u++) {
            float bu = indicator(cur_op, unary_ops[u]);
            out[S_AD_LEFT_GRAD_NEW] += bu * gsv;
        }
    }
}

/* Layer 4: AD tape write (forward ops) + gradient write-back (backward).
 *
 * Forward: when AD_IS_FORWARD is set, write AD_CUR_* fields into
 * tape[tape_len] and increment tape_len.
 *
 * Backward: when AD_IS_BACKWARD is set, accumulate gradient deltas
 * from AD_LEFT_GRAD_NEW / AD_RIGHT_GRAD_NEW into parent nodes,
 * then decrement AD_CURSOR. */
/* NOTE: layer4 takes original_state to distinguish "backward was already active"
 * from "backward was just started this cycle by layer3". */
/** @brief Simulated Layer 4 (tape write + parent load, gated FFN): records
 *         new AD tape node fields and heap/arena cons-cell writes
 *         (pair/vector/vec-elem/closure kinds) produced by Layer 3. */
static void layer4_ffn(float x[D], float out[D]) {
    memset(out, 0, D * sizeof(float));

    /* ── Forward: write new tape node ── */
    float fw = x[S_AD_IS_FORWARD];
    if (fw > 0.5f) {
        int tlen = (int)x[S_AD_TAPE_LEN];
        if (tlen >= 0 && tlen < AD_MAX_TAPE) {
            /* Write AD_CUR_* fields into tape[tlen] */
            for (int i = 0; i < AD_MAX_TAPE; i++) {
                float ti = indicator((float)tlen, (float)i);
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_OP]    += ti * x[S_AD_CUR_OP];
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE]  += ti * x[S_AD_CUR_VALUE];
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD]   += ti * x[S_AD_CUR_GRAD];
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_LEFT]   += ti * x[S_AD_CUR_LEFT];
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_RIGHT]  += ti * x[S_AD_CUR_RIGHT];
                out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_SAVED]  += ti * x[S_AD_CUR_SAVED];
            }
            out[S_AD_TAPE_LEN] += 1.0f; /* increment tape length */
        }
    }

    /* ── Backward: parent load ── */
    float bw = x[S_AD_IS_BACKWARD];
    if (bw > 0.5f) {
        float left_idx = x[S_AD_CUR_LEFT];
        float right_idx = x[S_AD_CUR_RIGHT];
        for (int i = 0; i < AD_MAX_TAPE; i++) {
            float li = indicator(left_idx, (float)i);
            float ri = indicator(right_idx, (float)i);
            out[S_AD_LEFT_VALUE]  += li * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
            out[S_AD_RIGHT_VALUE] += ri * x[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE];
        }
    }

    /* ── Arena pair read/write ──
     * Layer 3 computes the target cell and operation flags; Layer 4 performs
     * the actual bounded random access against the in-state arena bank. */
    float target = x[S_ARENA_TARGET];
    float wk = x[S_ARENA_WRITE_KIND];
    float wc = x[S_ARENA_WRITE_CAR];
    float wd = x[S_ARENA_WRITE_CDR];
    float rc = x[S_ARENA_READ_CAR];
    float rd = x[S_ARENA_READ_CDR];
    for (int cell = 0; cell < ARENA_CELLS; cell++) {
        float ti = indicator(target, (float)cell);
        int kind_dim = ARENA_DIM(cell, ARENA_F_KIND);
        int car_dim = ARENA_DIM(cell, ARENA_F_CAR_VAL);
        int cdr_dim = ARENA_DIM(cell, ARENA_F_CDR_VAL);
        int car_type_dim = ARENA_DIM(cell, ARENA_F_CAR_TYPE);
        int cdr_type_dim = ARENA_DIM(cell, ARENA_F_CDR_TYPE);

        out[kind_dim] += ti * wk * (x[S_ARENA_NEW_KIND] - x[kind_dim]);
        out[car_dim] += ti * wc * (x[S_ARENA_NEW_CAR] - x[car_dim]);
        out[car_type_dim] += ti * wc * (x[S_ARENA_NEW_CAR_TYPE] - x[car_type_dim]);
        out[cdr_dim] += ti * wd * (x[S_ARENA_NEW_CDR] - x[cdr_dim]);
        out[cdr_type_dim] += ti * wd * (x[S_ARENA_NEW_CDR_TYPE] - x[cdr_type_dim]);

        out[S_TOS] += ti * rc * (x[car_dim] - x[S_TOS]);
        out[S_TYPE_TOS] += ti * rc * (x[car_type_dim] - x[S_TYPE_TOS]);
        out[S_TOS] += ti * rd * (x[cdr_dim] - x[S_TOS]);
        out[S_TYPE_TOS] += ti * rd * (x[cdr_type_dim] - x[S_TYPE_TOS]);
    }

    /* ── Bounded continuation restore ──
     * Layer 3 marks INVOKE_CC by setting S_ARENA_VEC_HAS_E0 with a sentinel in
     * S_ARENA_VEC_LEN. The base cell is in S_ARENA_VEC_BASE; the record layout
     * mirrors OP_CALLCC's four arena cells. */
    float cont_restore = x[S_ARENA_VEC_HAS_E0] * indicator(x[S_ARENA_VEC_LEN], CONT_RESTORE_MARKER);
    if (cont_restore > 0.5f) {
        float rec[ARENA_CONT_CELLS][ARENA_CELL_FIELDS];
        memset(rec, 0, sizeof(rec));
        float base = x[S_ARENA_VEC_BASE];
        for (int off = 0; off < ARENA_CONT_CELLS; off++) {
            for (int cell = 0; cell < ARENA_CELLS; cell++) {
                float ti = indicator(base + (float)off, (float)cell);
                for (int f = 0; f < ARENA_CELL_FIELDS; f++)
                    rec[off][f] += ti * x[ARENA_DIM(cell, f)];
            }
        }
        out[S_PC] += cont_restore * (rec[0][ARENA_F_CAR_VAL] - x[S_PC]);
        out[S_DEPTH] += cont_restore * (rec[0][ARENA_F_CDR_VAL] + 1.0f - x[S_DEPTH]);
        out[S_MEM0] += cont_restore * (rec[2][ARENA_F_CAR_TYPE] - x[S_MEM0]);
        out[S_MEM1] += cont_restore * (rec[2][ARENA_F_CDR_TYPE] - x[S_MEM1]);
        out[S_MEM2] += cont_restore * (rec[3][ARENA_F_CAR_VAL] - x[S_MEM2]);
        out[S_MEM3] += cont_restore * (rec[3][ARENA_F_CDR_VAL] - x[S_MEM3]);
        out[S_SOS] += cont_restore * (rec[0][ARENA_F_CAR_TYPE] - x[S_SOS]);
        out[S_R2] += cont_restore * (rec[0][ARENA_F_CDR_TYPE] - x[S_R2]);
        out[S_R3] += cont_restore * (rec[1][ARENA_F_CAR_VAL] - x[S_R3]);
        out[S_TYPE_SOS] += cont_restore * (rec[1][ARENA_F_CAR_TYPE] - x[S_TYPE_SOS]);
        out[S_TYPE_R2] += cont_restore * (rec[1][ARENA_F_CDR_TYPE] - x[S_TYPE_R2]);
        out[S_TYPE_R3] += cont_restore * (rec[2][ARENA_F_CAR_VAL] - x[S_TYPE_R3]);
        out[S_WIND_DEPTH] += cont_restore * (rec[3][ARENA_F_CAR_TYPE] - x[S_WIND_DEPTH]);
    }

    /* ── Arena vector create ──
     * Vectors are a header cell followed by contiguous element cells:
     *   header.car = length
     *   elem[i].car = element value
     * VEC_REF/VEC_SET/VEC_LEN reuse the generic car read/write path above. */
    float vw = x[S_ARENA_VEC_WRITE];
    if (vw > 0.5f) {
        int elem_dims[4] = { S_ARENA_VEC_E0, S_ARENA_VEC_E1, S_ARENA_VEC_E2, S_ARENA_VEC_E3 };
        int elem_type_dims[4] = { S_ARENA_VEC_T0, S_ARENA_VEC_T1, S_ARENA_VEC_T2, S_ARENA_VEC_T3 };
        int elem_has_dims[4] = { S_ARENA_VEC_HAS_E0, S_ARENA_VEC_HAS_E1, S_ARENA_VEC_HAS_E2, S_ARENA_VEC_HAS_E3 };
        float base = x[S_ARENA_VEC_BASE];
        for (int cell = 0; cell < ARENA_CELLS; cell++) {
            int kind_dim = ARENA_DIM(cell, ARENA_F_KIND);
            int car_dim = ARENA_DIM(cell, ARENA_F_CAR_VAL);
            int cdr_dim = ARENA_DIM(cell, ARENA_F_CDR_VAL);
            int car_type_dim = ARENA_DIM(cell, ARENA_F_CAR_TYPE);
            int cdr_type_dim = ARENA_DIM(cell, ARENA_F_CDR_TYPE);
            float hi = indicator(base, (float)cell) * vw;
            out[kind_dim] += hi * (ARENA_KIND_VECTOR - x[kind_dim]);
            out[car_dim] += hi * (x[S_ARENA_VEC_LEN] - x[car_dim]);
            out[cdr_dim] += hi * ((base + 1.0f) - x[cdr_dim]);
            out[car_type_dim] += hi * (TYPE_NUMBER - x[car_type_dim]);
            out[cdr_type_dim] += hi * (TYPE_NUMBER - x[cdr_type_dim]);

            for (int i = 0; i < ARENA_MAX_INLINE_VECTOR; i++) {
                float ei = indicator(base + 1.0f + (float)i, (float)cell) * x[elem_has_dims[i]];
                out[kind_dim] += ei * (ARENA_KIND_VEC_ELEM - x[kind_dim]);
                out[car_dim] += ei * (x[elem_dims[i]] - x[car_dim]);
                out[cdr_dim] += ei * ((base + 2.0f + (float)i) - x[cdr_dim]);
                out[car_type_dim] += ei * (x[elem_type_dims[i]] - x[car_type_dim]);
                out[cdr_type_dim] += ei * (TYPE_NUMBER - x[cdr_type_dim]);
            }
        }
    }

    /* ── Arena list create for PACK_REST ──
     * Layer 3 precomputes the contiguous pair cells, including cdr links. */
    {
        int elem_dims[4] = { S_ARENA_LIST_E0, S_ARENA_LIST_E1, S_ARENA_LIST_E2, S_ARENA_LIST_E3 };
        int elem_type_dims[4] = { S_ARENA_LIST_T0, S_ARENA_LIST_T1, S_ARENA_LIST_T2, S_ARENA_LIST_T3 };
        int cdr_dims[4] = { S_ARENA_LIST_CDR0, S_ARENA_LIST_CDR1, S_ARENA_LIST_CDR2, S_ARENA_LIST_CDR3 };
        int cdr_type_dims[4] = { S_ARENA_LIST_CDRT0, S_ARENA_LIST_CDRT1, S_ARENA_LIST_CDRT2, S_ARENA_LIST_CDRT3 };
        int elem_has_dims[4] = { S_ARENA_LIST_HAS_E0, S_ARENA_LIST_HAS_E1, S_ARENA_LIST_HAS_E2, S_ARENA_LIST_HAS_E3 };
        float base = x[S_ARENA_LIST_BASE];
        for (int cell = 0; cell < ARENA_CELLS; cell++) {
            int kind_dim = ARENA_DIM(cell, ARENA_F_KIND);
            int car_dim = ARENA_DIM(cell, ARENA_F_CAR_VAL);
            int cdr_dim = ARENA_DIM(cell, ARENA_F_CDR_VAL);
            int car_type_dim = ARENA_DIM(cell, ARENA_F_CAR_TYPE);
            int cdr_type_dim = ARENA_DIM(cell, ARENA_F_CDR_TYPE);
            for (int i = 0; i < ARENA_MAX_INLINE_VECTOR; i++) {
                float ei = indicator(base + (float)i, (float)cell) * x[elem_has_dims[i]];
                out[kind_dim] += ei * (ARENA_KIND_PAIR - x[kind_dim]);
                out[car_dim] += ei * (x[elem_dims[i]] - x[car_dim]);
                out[cdr_dim] += ei * (x[cdr_dims[i]] - x[cdr_dim]);
                out[car_type_dim] += ei * (x[elem_type_dims[i]] - x[car_type_dim]);
                out[cdr_type_dim] += ei * (x[cdr_type_dims[i]] - x[cdr_type_dim]);
            }
        }
    }
}

/* Layer 5 simulated: backward gradient dispatch + write-back.
 * This is the COMPLETE backward computation layer. Layer 3 is NEVER invoked
 * during backward — this eliminates all transient-clearing interference.
 *
 * Gradient rules dispatch on AD_CUR_OP:
 *   Binary: ADD (passthrough), SUB (negate right), MUL (cross-product via Layer 2 SQUARE)
 *   Unary: ALL use grad * saved_val (precomputed as AD_PROD_GRAD_SV by Layer 2 SQUARE)
 *
 * After computing LEFT/RIGHT_GRAD_NEW, writes them to parent tape nodes. */
/* Layer 5 simulated: split into dispatch and write-back for the two-pass scheme. */
/** @brief Simulated Layer 5, pass 1 (backward-only gated FFN): during an
 *         active AD backward pass, dispatches on the current tape node's op
 *         type to compute the raw gradient contributions to propagate to
 *         its parent nodes. */
static void layer5_dispatch(float x[D], float out[D]) {
    memset(out, 0, D * sizeof(float));
    float bw = x[S_AD_IS_BACKWARD];
    if (bw > 0.5f) {
        float cur_op = x[S_AD_CUR_OP];
        float cur_grad = x[S_AD_CUR_GRAD];
        float gl = x[S_AD_PROD_GRAD_LV];
        float gr = x[S_AD_PROD_GRAD_RV];
        float gsv = x[S_AD_PROD_GRAD_SV];

        /* Binary ops */
        float ba = indicator(cur_op, AD_OP_ADD);
        out[S_AD_LEFT_GRAD_NEW] += ba * cur_grad;
        out[S_AD_RIGHT_GRAD_NEW] += ba * cur_grad;
        float bs = indicator(cur_op, AD_OP_SUB);
        out[S_AD_LEFT_GRAD_NEW] += bs * cur_grad;
        out[S_AD_RIGHT_GRAD_NEW] -= bs * cur_grad;
        float bm = indicator(cur_op, AD_OP_MUL);
        out[S_AD_LEFT_GRAD_NEW] += bm * gl;
        out[S_AD_RIGHT_GRAD_NEW] += bm * gr;
        float bd = indicator(cur_op, AD_OP_DIV);
        float safe_div_right = fabsf(x[S_AD_RIGHT_VALUE]) > 1e-15f ? x[S_AD_RIGHT_VALUE] :
                               (x[S_AD_RIGHT_VALUE] < 0 ? -1e-15f : 1e-15f);
        out[S_AD_LEFT_GRAD_NEW] += bd * gsv;
        out[S_AD_RIGHT_GRAD_NEW] += bd * gr * (-1.0f / (safe_div_right * safe_div_right));
        float bp = indicator(cur_op, AD_OP_POW);
        float safe_pow_left = x[S_AD_LEFT_VALUE] > 1e-15f ? x[S_AD_LEFT_VALUE] : 1e-15f;
        out[S_AD_LEFT_GRAD_NEW] += bp * gsv;
        out[S_AD_RIGHT_GRAD_NEW] += bp * cur_grad * x[S_AD_CUR_VALUE] * logf(safe_pow_left);

        /* ALL unary ops: dL = grad * saved_val */
        float unary_ops[] = { AD_OP_NEG, AD_OP_ABS, AD_OP_RELU, AD_OP_SIGMOID,
                              AD_OP_TANH, AD_OP_EXP, AD_OP_LOG, AD_OP_SQRT,
                              AD_OP_SIN, AD_OP_COS };
        for (int u = 0; u < 10; u++)
            out[S_AD_LEFT_GRAD_NEW] += indicator(cur_op, unary_ops[u]) * gsv;
    }
}
/** @brief Simulated Layer 5, pass 2 (backward-only gated FFN): writes the
 *         gradient contributions computed by layer5_dispatch() back into
 *         the parent tape nodes' AD_F_GRAD fields and advances the AD
 *         cursor. */
static void layer5_writeback(float x[D], float out[D]) {
    memset(out, 0, D * sizeof(float));
    float bw = x[S_AD_IS_BACKWARD];
    if (bw > 0.5f) {
        float left_idx = x[S_AD_CUR_LEFT];
        float right_idx = x[S_AD_CUR_RIGHT];
        float lg = x[S_AD_LEFT_GRAD_NEW];
        float rg = x[S_AD_RIGHT_GRAD_NEW];
        for (int i = 0; i < AD_MAX_TAPE; i++) {
            float li = indicator(left_idx, (float)i);
            float ri = indicator(right_idx, (float)i);
            out[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_GRAD] += li * lg + ri * rg;
        }
    }
}
/* layer5_ffn for the weight-matrix path calls apply_ffn_layer(w,5,x) twice.
 * The simulated path calls layer5_dispatch then layer5_writeback instead. */
/** @brief Unused placeholder for a combined Layer 5 FFN; the simulated
 *         execution path calls layer5_dispatch() and layer5_writeback()
 *         separately instead. */
static void layer5_ffn(float x[D], float out[D]) {
    /* Not used directly — sim path calls dispatch+writeback separately */
    (void)x; (void)out;
}

/* Post-process: handle IS_NATIVE, IS_CALL, IS_RET flags.
 * The transformer set flags and PC++. The exec loop performs the actual operations. */
/**
 * @brief Post-process a state vector after one execute_step()/simulated
 *        layer pass: handles opcodes flagged IS_NATIVE (DIV, MOD, and other
 *        ops that don't fit the pure gated-FFN weight computation and are
 *        instead finished here in plain C), then clears IS_NATIVE.
 */
