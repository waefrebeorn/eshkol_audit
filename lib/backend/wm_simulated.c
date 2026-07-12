static int run_simulated(const Instr* prog, int n_instr, float* outputs, int max_out) {
    /* pe is zero-initialised so out-of-range positions (PC ≥ n_instr) attend
     * to an all-zero embedding (opcode = OP_NOP), avoiding garbage attention
     * scores that would otherwise dispatch arbitrary instructions. The
     * reference VM's auto-halt on `pc >= n_instr` is the canonical
     * out-of-bounds behaviour; the matrix path emulates this by ensuring
     * those positions have a predictable, well-defined embedding. */
    float pe[256][D];
    memset(pe, 0, sizeof(pe));
    for (int p = 0; p < n_instr && p < 256; p++) embed_instruction(&prog[p], p, pe[p]);
    float state[D]; memset(state, 0, sizeof(state)); state[S_OUTPUT] = -1; state[S_CUR_CLOSURE] = -100.0f;
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int n_out = 0, step_count = 0;
    int sim_trace_step_cap = g_last_ref_steps > 0 ? g_last_ref_steps : 8192;
    emit_trace_line(g_trace_sim_fp, 0, state, prog, n_instr, -1);
    for (int step = 0; step < 8192; step++) {
        step_count++;
        float x[D]; memcpy(x, state, sizeof(x));
        float tmp[D];

        /* Clear transient dims at start of each cycle (matches execute_step).
         * S_AD_IS_BACKWARD (113) and S_AD_MODE (38) persist — checked by loop. */
        for (int i = S_AD_CUR_OP; i <= S_AD_RIGHT_VALUE; i++) x[i] = 0;
        x[S_AD_IS_FORWARD] = 0; /* 112 */
        /* Skip S_AD_IS_BACKWARD (113) — must persist for backward check */
        for (int i = S_AD_GRAD_ACCUM; i <= S_AD_SPARE8; i++) x[i] = 0;
        for (int i = S_ARENA_TRANSIENT_START; i <= S_ARENA_TRANSIENT_END; i++) x[i] = 0;

        if (x[S_AD_IS_BACKWARD] > 0.5f) {
            /* Backward: simulated layer functions mirroring backward_with_weights.
             * Uses same layer math as weight matrices: indicator gating, SQUARE products,
             * gradient dispatch — expressed as explicit C simulating the neuron computations. */
            /* Backward: L1→L4→L2→L5 (Layer 3 NOT invoked during backward).
             * L1 loads cursor node, L4 loads parent values, L2 computes
             * SQUARE products, L5 dispatches gradient rules + writes back. */
            /* Backward: L1→L4→L2→L5→L5 through simulated layers.
             * Layer 5 handles gradient dispatch, write-back, cursor decrement,
             * completion check, and transient clearing — no inline C. */
            layer2_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer4_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer1_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer5_dispatch(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer5_writeback(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            /* Cursor decrement, completion check, and transient clear are
             * in layer2_ffn (weight neurons). No inline C in backward loop. */
        } else {
            /* Normal execution: fetch, preprocess, product, dispatch, writeback. */
            layer0_attention(x, pe, n_instr, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer2_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer1_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer3_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            layer4_ffn(x, tmp); for(int i=0;i<D;i++) x[i]+=tmp[i];
            int sim_is_native = x[S_IS_NATIVE] > 0.5f ? 1 : 0;
            exec_loop_postprocess(x, prog, n_instr);
            (void)sim_is_native; /* captured but unused below — keep for parity */
        }
        if (x[S_HAS_OUT] > 0.5f && n_out < max_out) outputs[n_out++] = x[S_OUTPUT];
        if (x[S_HALT] > 0.5f) {
            memcpy(state, x, sizeof(state));
            if (step_count <= sim_trace_step_cap)
                emit_trace_line(g_trace_sim_fp, step_count, state, prog, n_instr, -1);
            break;
        }
        memcpy(state, x, sizeof(state));
        if (step_count <= sim_trace_step_cap)
            emit_trace_line(g_trace_sim_fp, step_count, state, prog, n_instr, -1);
    }
    g_last_sim_steps = step_count;
    return n_out;
}

/*******************************************************************************
 * 3. Explicit Weight Matrices + Matrix-Based Forward Pass
 ******************************************************************************/

typedef struct {
    float wq[N_LAYERS][D * D];
    float wk[N_LAYERS][D * D];
    float wv[N_LAYERS][D * D];
    float wo[N_LAYERS][D * D];
    float bq[N_LAYERS][D];
    float ff_up[N_LAYERS][D * FFN_DIM];
    float ff_up_b[N_LAYERS][FFN_DIM];
    float ff_down[N_LAYERS][FFN_DIM * D];
    float ff_down_b[N_LAYERS][D];
    float ff_gate[N_LAYERS][D * FFN_DIM];
    float ff_gate_b[N_LAYERS][FFN_DIM];
    int   ff_type[N_LAYERS];  /* 0=noop, 1=standard+square, 2=gated+sigmoid */
} InterpreterWeights;

#define W(mat, r, c, cols) ((mat)[(r) * (cols) + (c)])

/**
 * @brief Add a gated neuron pair for opcode dispatch to weight layer @p L
 *        starting at hidden-unit index @p n.
 *
 * Implements indicator(opcode==op_id) * alive * linear_combination ->
 * out_dim, encoded as two sigmoid-gated FFN neurons (a "+coeff"/"-coeff"
 * pair) whose difference reproduces the indicator() gate.
 * @return The next free hidden-unit index (n + 2).
 */
static int add_gated_pair(InterpreterWeights* w, int L, int n,
                           int op_id,
                           int ud1, float us1, int ud2, float us2,
                           int ud3, float us3, int ud4, float us4,
                           float ubias,
                           int out_dim, float coeff) {
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], S_OPCODE, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], S_HALT, j, FFN_DIM)   += -SCALE;
        w->ff_gate_b[L][j] = SCALE * (-(float)op_id + s);
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        float c = (sign == 0) ? coeff : -coeff;
        W(w->ff_down[L], j, out_dim, D) += c;
    }
    return n + 2;
}

/** @brief Like add_gated_pair(), but the gate additionally dispatches on
 *         one bounded integer state dimension equalling @p index (opcode
 *         and index are packed into a single scaled target so one gate
 *         pair covers both conditions). */
static int add_gated_opcode_index(InterpreterWeights* w, int L, int n,
                                  int op_id, int index_dim, int index,
                                  int ud1, float us1, int ud2, float us2,
                                  int ud3, float us3, int ud4, float us4,
                                  float ubias,
                                  int out_dim, float coeff) {
    const float index_scale = 100.0f;
    float target = (float)op_id + index_scale * (float)index;
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], S_OPCODE, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], index_dim, j, FFN_DIM) += SCALE * index_scale;
        W(w->ff_gate[L], S_HALT, j, FFN_DIM) += -SCALE;
        w->ff_gate_b[L][j] = SCALE * (-target + s);
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        W(w->ff_down[L], j, out_dim, D) += (sign == 0) ? coeff : -coeff;
    }
    return n + 2;
}

/** @brief Like add_gated_opcode_index(), but the gate dispatches on the
 *         opcode plus two bounded integer state dimensions, packed into one
 *         scaled target. */
static int add_gated_opcode_two_indices(InterpreterWeights* w, int L, int n,
                                        int op_id,
                                        int index_dim1, int index1,
                                        int index_dim2, int index2,
                                        int ud1, float us1, int ud2, float us2,
                                        int ud3, float us3, int ud4, float us4,
                                        float ubias,
                                        int out_dim, float coeff) {
    const float index1_scale = 100.0f;
    const float index2_scale = 10000.0f;
    float target = (float)op_id + index1_scale * (float)index1 +
                   index2_scale * (float)index2;
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], S_OPCODE, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], index_dim1, j, FFN_DIM) += SCALE * index1_scale;
        W(w->ff_gate[L], index_dim2, j, FFN_DIM) += SCALE * index2_scale;
        W(w->ff_gate[L], S_HALT, j, FFN_DIM) += -SCALE;
        w->ff_gate_b[L][j] = SCALE * (-target + s);
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        W(w->ff_down[L], j, out_dim, D) += (sign == 0) ? coeff : -coeff;
    }
    return n + 2;
}

/** @brief Same as add_gated_pair(), but additionally suppresses the gate
 *         while an AD backward pass is active (S_AD_IS_BACKWARD), for
 *         opcodes whose forward-only semantics must not fire mid-backward. */
static int add_gated_pair_ad(InterpreterWeights* w, int L, int n,
                              int op_id,
                              int ud1, float us1, int ud2, float us2,
                              int ud3, float us3, int ud4, float us4,
                              float ubias,
                              int out_dim, float coeff) {
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], S_OPCODE, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], S_HALT, j, FFN_DIM)   += -SCALE;
        W(w->ff_gate[L], S_AD_IS_BACKWARD, j, FFN_DIM) += -SCALE; /* suppress during backward */
        w->ff_gate_b[L][j] = SCALE * (-(float)op_id + s);
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        float c = (sign == 0) ? coeff : -coeff;
        W(w->ff_down[L], j, out_dim, D) += c;
    }
    return n + 2;
}

/** @brief Same as add_gated_pair_ad(), but the gate also dispatches on a
 *         bounded stack index (TOS/SOS == @p slot). Used for forward tape
 *         random-access reads and AD_GRAD. */
static int add_gated_pair_ad_index(InterpreterWeights* w, int L, int n,
                                   int op_id, int index_dim, int slot,
                                   int ud1, float us1, int ud2, float us2,
                                   int ud3, float us3, int ud4, float us4,
                                   float ubias,
                                   int out_dim, float coeff) {
    const float index_scale = 100.0f;
    float target = (float)op_id + index_scale * (float)slot;
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], S_OPCODE, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], index_dim, j, FFN_DIM) += SCALE * index_scale;
        W(w->ff_gate[L], S_HALT, j, FFN_DIM) += -SCALE;
        W(w->ff_gate[L], S_AD_IS_BACKWARD, j, FFN_DIM) += -SCALE;
        w->ff_gate_b[L][j] = SCALE * (-target + s);
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        W(w->ff_down[L], j, out_dim, D) += (sign == 0) ? coeff : -coeff;
    }
    return n + 2;
}

/** @brief Add a single one-sided value gate controlled by a precomputed
 *         binary flag: gate ~= flag && (value_sign * value_dim >
 *         threshold).
 * @return The next free hidden-unit index (n + 1).
 */
static int add_flagged_value_halfspace(InterpreterWeights* w, int L, int n,
                                       int flag_dim, int value_dim,
                                       float value_sign, float threshold,
                                       int ud1, float us1, int ud2, float us2,
                                       int ud3, float us3, int ud4, float us4,
                                       float ubias,
                                       int out_dim, float coeff) {
    const float flag_scale = 1000.0f;
    W(w->ff_gate[L], flag_dim, n, FFN_DIM) = flag_scale * SCALE;
    W(w->ff_gate[L], value_dim, n, FFN_DIM) = value_sign * SCALE;
    W(w->ff_gate[L], S_HALT, n, FFN_DIM) = -flag_scale * SCALE;
    w->ff_gate_b[L][n] = -flag_scale * SCALE - threshold * SCALE;
    if (ud1 >= 0) W(w->ff_up[L], ud1, n, FFN_DIM) += us1;
    if (ud2 >= 0) W(w->ff_up[L], ud2, n, FFN_DIM) += us2;
    if (ud3 >= 0) W(w->ff_up[L], ud3, n, FFN_DIM) += us3;
    if (ud4 >= 0) W(w->ff_up[L], ud4, n, FFN_DIM) += us4;
    w->ff_up_b[L][n] = ubias;
    W(w->ff_down[L], n, out_dim, D) += coeff;
    return n + 1;
}

/** @brief Add a single saturated gate for a precomputed binary flag.
 *         Unlike the opcode-difference pair helpers, this does not add
 *         far-opcode cancellation residue.
 * @return The next free hidden-unit index (n + 1).
 */
static int add_flagged_linear(InterpreterWeights* w, int L, int n,
                              int flag_dim,
                              int ud1, float us1, int ud2, float us2,
                              int ud3, float us3, int ud4, float us4,
                              float ubias,
                              int out_dim, float coeff) {
    W(w->ff_gate[L], flag_dim, n, FFN_DIM) = SCALE;
    W(w->ff_gate[L], S_HALT, n, FFN_DIM) = -SCALE;
    w->ff_gate_b[L][n] = -0.5f * SCALE;
    if (ud1 >= 0) W(w->ff_up[L], ud1, n, FFN_DIM) += us1;
    if (ud2 >= 0) W(w->ff_up[L], ud2, n, FFN_DIM) += us2;
    if (ud3 >= 0) W(w->ff_up[L], ud3, n, FFN_DIM) += us3;
    if (ud4 >= 0) W(w->ff_up[L], ud4, n, FFN_DIM) += us4;
    w->ff_up_b[L][n] = ubias;
    W(w->ff_down[L], n, out_dim, D) += coeff;
    return n + 1;
}

/** @brief Add a single always-on (unconditional) hidden neuron, gated only
 *         by "not halted".
 * @return The next free hidden-unit index (n + 1).
 */
static int add_unconditional(InterpreterWeights* w, int L, int n,
                              int ud1, float us1, float ubias,
                              int out_dim, float coeff) {
    int j = n;
    W(w->ff_gate[L], S_HALT, j, FFN_DIM) = -SCALE;
    w->ff_gate_b[L][j] = 10.0f * SCALE;
    if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) = us1;
    w->ff_up_b[L][j] = ubias;
    W(w->ff_down[L], j, out_dim, D) = coeff;
    return n + 1;
}

/** @brief Add two hidden neurons that jointly compute
 *         indicator(state_dim == target_value) into @p out_dim.
 * @return The next free hidden-unit index (n + 2).
 */
static int add_indicator_precompute(InterpreterWeights* w, int L, int n,
                                    int state_dim, float target_value,
                                    int out_dim) {
    W(w->ff_gate[L], state_dim, n, FFN_DIM) = SCALE;
    w->ff_gate_b[L][n] = SCALE * (-target_value + 0.5f);
    w->ff_up_b[L][n] = 1.0f;
    W(w->ff_down[L], n, out_dim, D) = 1.0f;
    n++;

    W(w->ff_gate[L], state_dim, n, FFN_DIM) = SCALE;
    w->ff_gate_b[L][n] = SCALE * (-target_value - 0.5f);
    w->ff_up_b[L][n] = 1.0f;
    W(w->ff_down[L], n, out_dim, D) = -1.0f;
    return n + 1;
}

/**
 * @brief Add a gated pair for fixed small-immediate opcodes such as POPN.
 *        The gate is indicator(opcode + 100*operand == op_id +
 *        100*operand_id), collision-free for this ISA's opcode and small
 *        operand ranges.
 * @return The next free hidden-unit index (n + 2).
 */
static int add_gated_pair_op_operand(InterpreterWeights* w, int L, int n,
                                     int op_id, int operand_id,
                                     int ud1, float us1, int ud2, float us2,
                                     int ud3, float us3, int ud4, float us4,
                                     float ubias,
                                     int out_dim, float coeff) {
    const float operand_scale = 100.0f;
    float target = (float)op_id + operand_scale * (float)operand_id;
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], S_OPCODE, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], S_OPERAND, j, FFN_DIM) += SCALE * operand_scale;
        W(w->ff_gate[L], S_HALT, j, FFN_DIM) += -SCALE;
        w->ff_gate_b[L][j] = SCALE * (-target + s);
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        float c = (sign == 0) ? coeff : -coeff;
        W(w->ff_down[L], j, out_dim, D) += c;
    }
    return n + 2;
}

/**
 * @brief Add a gated pair conditioned on an arena operation flag AND the
 *        arena target cell. When @p flag_dim is 1, the pair difference is
 *        indicator(A_TARGET == cell); when 0, the large negative flag bias
 *        keeps both neurons closed for all bounded arena targets.
 * @return The next free hidden-unit index (n + 2).
 */
static int add_arena_target_pair(InterpreterWeights* w, int L, int n,
                                 int flag_dim, int cell,
                                 int ud1, float us1, int ud2, float us2,
                                 int ud3, float us3, int ud4, float us4,
                                 float ubias,
                                 int out_dim, float coeff) {
    const float flag_scale = 100.0f;
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], S_ARENA_TARGET, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], flag_dim, j, FFN_DIM) += flag_scale * SCALE;
        w->ff_gate_b[L][j] = SCALE * (-(float)cell + s) - flag_scale * SCALE;
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        W(w->ff_down[L], j, out_dim, D) += (sign == 0) ? coeff : -coeff;
    }
    return n + 2;
}

/** @brief Add a gated pair conditioned on a vector-create flag AND (arena
 *         vector base + @p offset == @p cell); returns @p n unchanged (no
 *         neurons added) if the implied base target falls outside
 *         [0, ARENA_CELLS).
 * @return The next free hidden-unit index (n + 2, or n if out of range).
 */
static int add_arena_vec_offset_pair(InterpreterWeights* w, int L, int n,
                                     int flag_dim, int cell, int offset,
                                     int ud1, float us1, int ud2, float us2,
                                     int ud3, float us3, int ud4, float us4,
                                     float ubias,
                                     int out_dim, float coeff) {
    int base_target = cell - offset;
    if (base_target < 0 || base_target >= ARENA_CELLS) return n;
    const float flag_scale = 100.0f;
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], S_ARENA_VEC_BASE, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], flag_dim, j, FFN_DIM) += flag_scale * SCALE;
        w->ff_gate_b[L][j] = SCALE * (-(float)base_target + s) - flag_scale * SCALE;
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        W(w->ff_down[L], j, out_dim, D) += (sign == 0) ? coeff : -coeff;
    }
    return n + 2;
}

/** @brief Same as add_arena_vec_offset_pair(), but the base state dimension
 *         is selectable via @p base_dim rather than fixed to
 *         S_ARENA_VEC_BASE. */
static int add_arena_base_offset_pair(InterpreterWeights* w, int L, int n,
                                      int base_dim, int flag_dim, int cell, int offset,
                                      int ud1, float us1, int ud2, float us2,
                                      int ud3, float us3, int ud4, float us4,
                                      float ubias,
                                      int out_dim, float coeff) {
    int base_target = cell - offset;
    if (base_target < 0 || base_target >= ARENA_CELLS) return n;
    const float flag_scale = 100.0f;
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], base_dim, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], flag_dim, j, FFN_DIM) += flag_scale * SCALE;
        w->ff_gate_b[L][j] = SCALE * (-(float)base_target + s) - flag_scale * SCALE;
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        W(w->ff_down[L], j, out_dim, D) += (sign == 0) ? coeff : -coeff;
    }
    return n + 2;
}

/**
 * @brief Same as add_arena_base_offset_pair(), plus an additional sentinel
 *        marker condition. Used by INVOKE_CC restore so it can reuse
 *        existing arena transient lanes without colliding with vector
 *        creation or PACK_REST writes.
 */
static int add_arena_marked_base_offset_pair(InterpreterWeights* w, int L, int n,
                                             int base_dim, int flag_dim,
                                             int marker_dim, float marker_value,
                                             int cell, int offset,
                                             int ud1, float us1, int ud2, float us2,
                                             int ud3, float us3, int ud4, float us4,
                                             float ubias,
                                             int out_dim, float coeff) {
    int base_target = cell - offset;
    if (base_target < 0 || base_target >= ARENA_CELLS) return n;
    const float flag_scale = 100.0f;
    const float marker_scale = 10000.0f;
    for (int sign = 0; sign < 2; sign++) {
        int j = n + sign;
        float s = (sign == 0) ? 0.5f : -0.5f;
        W(w->ff_gate[L], base_dim, j, FFN_DIM) += SCALE;
        W(w->ff_gate[L], flag_dim, j, FFN_DIM) += flag_scale * SCALE;
        W(w->ff_gate[L], marker_dim, j, FFN_DIM) += marker_scale * SCALE;
        w->ff_gate_b[L][j] = SCALE * (-(float)base_target + s)
                            - flag_scale * SCALE
                            - marker_scale * marker_value * SCALE;
        if (ud1 >= 0) W(w->ff_up[L], ud1, j, FFN_DIM) += us1;
        if (ud2 >= 0) W(w->ff_up[L], ud2, j, FFN_DIM) += us2;
        if (ud3 >= 0) W(w->ff_up[L], ud3, j, FFN_DIM) += us3;
        if (ud4 >= 0) W(w->ff_up[L], ud4, j, FFN_DIM) += us4;
        w->ff_up_b[L][j] = ubias;
        W(w->ff_down[L], j, out_dim, D) += (sign == 0) ? coeff : -coeff;
    }
    return n + 2;
}

/** @brief Add weights that shift the type-tag stack (TYPE_TOS/SOS/R2/R3)
 *         down by one on @p op_id, pushing @p pushed_type onto TYPE_TOS. */
static int add_type_push(InterpreterWeights* w, int L, int n,
                         int op_id, float pushed_type) {
    n = add_gated_pair(w, L, n, op_id, S_TYPE_TOS,-1,-1,0,-1,0,-1,0,
                       pushed_type, S_TYPE_TOS, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_TOS,1,S_TYPE_SOS,-1,-1,0,-1,0,
                       0, S_TYPE_SOS, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_SOS,1,S_TYPE_R2,-1,-1,0,-1,0,
                       0, S_TYPE_R2, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R2,1,S_TYPE_R3,-1,-1,0,-1,0,
                       0, S_TYPE_R3, 1.0f);
    return n;
}

/** @brief Add weights for the fixed-@p count case of the vector-create
 *         opcode (op 39): writes the new arena vector's length/next-cell
 *         bookkeeping and copies up to ARENA_MAX_INLINE_VECTOR popped
 *         stack values (and their type tags) into the vector's inline
 *         element slots. */
static int add_vec_create_case(InterpreterWeights* w, int L, int n, int count) {
    int elem_dims[4] = { S_ARENA_VEC_E0, S_ARENA_VEC_E1, S_ARENA_VEC_E2, S_ARENA_VEC_E3 };
    int elem_type_dims[4] = { S_ARENA_VEC_T0, S_ARENA_VEC_T1, S_ARENA_VEC_T2, S_ARENA_VEC_T3 };
    int elem_has_dims[4] = { S_ARENA_VEC_HAS_E0, S_ARENA_VEC_HAS_E1, S_ARENA_VEC_HAS_E2, S_ARENA_VEC_HAS_E3 };
    int value_srcs[4] = { S_TOS, S_SOS, S_R2, S_R3 };
    int type_srcs[4] = { S_TYPE_TOS, S_TYPE_SOS, S_TYPE_R2, S_TYPE_R3 };

    n = add_gated_pair_op_operand(w,L,n, 39,count, -1,0,-1,0,-1,0,-1,0, 1.0f-(float)count, S_DEPTH, 1.0f);
    n = add_gated_pair_op_operand(w,L,n, 39,count, -1,0,-1,0,-1,0,-1,0, (float)count, S_ARENA_VEC_LEN, 1.0f);
    n = add_gated_pair_op_operand(w,L,n, 39,count, -1,0,-1,0,-1,0,-1,0, (float)(count + 1), S_ARENA_NEXT, 1.0f);

    for (int i = 0; i < count && i < ARENA_MAX_INLINE_VECTOR; i++) {
        int src = count - 1 - i;
        n = add_gated_pair_op_operand(w,L,n, 39,count, value_srcs[src],1,-1,0,-1,0,-1,0, 0, elem_dims[i], 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 39,count, type_srcs[src],1,-1,0,-1,0,-1,0, 0, elem_type_dims[i], 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 39,count, -1,0,-1,0,-1,0,-1,0, 1.0f, elem_has_dims[i], 1.0f);
    }

    return n;
}

/** @brief Add weights that shift the type-tag stack up by one on @p op_id
 *         (pop TYPE_TOS, backfilling from SOS/R2/R3 and defaulting R3 to
 *         TYPE_NUMBER). */
static int add_type_pop(InterpreterWeights* w, int L, int n, int op_id) {
    n = add_gated_pair(w, L, n, op_id, S_TYPE_SOS,1,S_TYPE_TOS,-1,-1,0,-1,0,
                       0, S_TYPE_TOS, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R2,1,S_TYPE_SOS,-1,-1,0,-1,0,
                       0, S_TYPE_SOS, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R3,1,S_TYPE_R2,-1,-1,0,-1,0,
                       0, S_TYPE_R2, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R3,-1,-1,0,-1,0,-1,0,
                       TYPE_NUMBER, S_TYPE_R3, 1.0f);
    return n;
}

/** @brief Add weights that duplicate TYPE_TOS's tag down the type-tag stack
 *         on @p op_id (mirrors a DUP opcode's effect on the value stack). */
static int add_type_dup(InterpreterWeights* w, int L, int n, int op_id) {
    n = add_gated_pair(w, L, n, op_id, S_TYPE_TOS,1,S_TYPE_SOS,-1,-1,0,-1,0,
                       0, S_TYPE_SOS, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_SOS,1,S_TYPE_R2,-1,-1,0,-1,0,
                       0, S_TYPE_R2, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R2,1,S_TYPE_R3,-1,-1,0,-1,0,
                       0, S_TYPE_R3, 1.0f);
    return n;
}

/* Exchange the type tags of TOS and SOS (R2/R3 type tags unchanged). */
static int add_type_swap(InterpreterWeights* w, int L, int n, int op_id) {
    n = add_gated_pair(w, L, n, op_id, S_TYPE_SOS,1,S_TYPE_TOS,-1,-1,0,-1,0,
                       0, S_TYPE_TOS, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_TOS,1,S_TYPE_SOS,-1,-1,0,-1,0,
                       0, S_TYPE_SOS, 1.0f);
    return n;
}

/** @brief Add weights for a binary-operator opcode's effect on the
 *         type-tag stack: replaces TYPE_TOS with @p result_type and shifts
 *         the rest of the tag stack up by one (consuming SOS's operand). */
static int add_type_binary_result(InterpreterWeights* w, int L, int n,
                                  int op_id, float result_type) {
    n = add_gated_pair(w, L, n, op_id, S_TYPE_TOS,-1,-1,0,-1,0,-1,0,
                       result_type, S_TYPE_TOS, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R2,1,S_TYPE_SOS,-1,-1,0,-1,0,
                       0, S_TYPE_SOS, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R3,1,S_TYPE_R2,-1,-1,0,-1,0,
                       0, S_TYPE_R2, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R3,-1,-1,0,-1,0,-1,0,
                       TYPE_NUMBER, S_TYPE_R3, 1.0f);
    return n;
}

/** @brief Add weights that pop two type tags off the tag stack on @p op_id
 *         (backfilling TOS/SOS from R2/R3, defaulting R2/R3 to
 *         TYPE_NUMBER). */
static int add_type_pop2(InterpreterWeights* w, int L, int n, int op_id) {
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R2,1,S_TYPE_TOS,-1,-1,0,-1,0,
                       0, S_TYPE_TOS, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R3,1,S_TYPE_SOS,-1,-1,0,-1,0,
                       0, S_TYPE_SOS, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R2,-1,-1,0,-1,0,-1,0,
                       TYPE_NUMBER, S_TYPE_R2, 1.0f);
    n = add_gated_pair(w, L, n, op_id, S_TYPE_R3,-1,-1,0,-1,0,-1,0,
                       TYPE_NUMBER, S_TYPE_R3, 1.0f);
    return n;
}

/** @brief Add weights that set TYPE_TOS to @p result_type on @p op_id (a
 *         unary operator's effect on the type-tag stack), leaving the rest
 *         of the tag stack unchanged. */
static int add_type_unary_result(InterpreterWeights* w, int L, int n,
                                 int op_id, float result_type) {
    return add_gated_pair(w, L, n, op_id, S_TYPE_TOS,-1,-1,0,-1,0,-1,0,
                          result_type, S_TYPE_TOS, 1.0f);
}

/**
 * @brief Build the full explicit weight matrices (Q/K/V/O attention
 *        projections and FFN up/gate/down matrices for all N_LAYERS) that
 *        encode the entire Eshkol VM ISA as a transformer forward pass.
 *
 * Zero-initializes @p w, then wires Layer 0's attention for instruction
 * fetch and populates each subsequent layer's gated FFN neurons — via the
 * add_gated_, add_arena_, and add_type_ helper families above — with one gated
 * neuron pair per opcode (and AD/arena/type-tag bookkeeping op) so that the
 * resulting matrix forward pass (forward_with_weights()) reproduces
 * execute_step()'s semantics bit-for-bit.
 */
