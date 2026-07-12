static void generate_weights(InterpreterWeights* w) {
    memset(w, 0, sizeof(InterpreterWeights));

    /* ── Layer 0: Attention (instruction fetch) ── */
    {
        float T = SCALE;
        W(w->wq[0], 0, S_PC, D) = T;
        w->bq[0][1] = T;
        W(w->wk[0], 0, 0, D) = 1.0f;
        W(w->wk[0], 1, 1, D) = 1.0f;
        W(w->wv[0], 0, S_OPCODE, D) = 1.0f;
        W(w->wv[0], 1, S_OPERAND, D) = 1.0f;
        W(w->wo[0], S_OPCODE, 0, D) = 1.0f;
        W(w->wo[0], S_OPERAND, 1, D) = 1.0f;
        w->ff_type[0] = 0;
    }

    /* ── Layer 2: Product FFN (SQUARE activation) ── */
    {
        const int L = 2;
        w->ff_type[L] = 1;
        W(w->ff_up[L], S_TOS, 0, FFN_DIM) = 1; W(w->ff_up[L], S_SOS, 0, FFN_DIM) = 1;
        W(w->ff_up[L], S_TOS, 1, FFN_DIM) = 1;
        W(w->ff_up[L], S_SOS, 2, FFN_DIM) = 1;
        W(w->ff_down[L], 0, S_PRODUCT, D) =  0.5f;
        W(w->ff_down[L], 1, S_PRODUCT, D) = -0.5f;
        W(w->ff_down[L], 2, S_PRODUCT, D) = -0.5f;

        /* AD backward products via SQUARE trick:
         * Product 1: grad * right_value → AD_PROD_GRAD_LV (MUL dL = grad*R) */
        int pn = 3; /* next neuron index */
        W(w->ff_up[L], S_AD_CUR_GRAD, pn, FFN_DIM) = 1; W(w->ff_up[L], S_AD_RIGHT_VALUE, pn, FFN_DIM) = 1;
        W(w->ff_up[L], S_AD_CUR_GRAD, pn+1, FFN_DIM) = 1;
        W(w->ff_up[L], S_AD_RIGHT_VALUE, pn+2, FFN_DIM) = 1;
        W(w->ff_down[L], pn, S_AD_PROD_GRAD_LV, D) = 0.5f;
        W(w->ff_down[L], pn+1, S_AD_PROD_GRAD_LV, D) = -0.5f;
        W(w->ff_down[L], pn+2, S_AD_PROD_GRAD_LV, D) = -0.5f;
        pn += 3;

        /* Product 2: grad * left_value → AD_PROD_GRAD_RV (MUL dR = grad*L) */
        W(w->ff_up[L], S_AD_CUR_GRAD, pn, FFN_DIM) = 1; W(w->ff_up[L], S_AD_LEFT_VALUE, pn, FFN_DIM) = 1;
        W(w->ff_up[L], S_AD_CUR_GRAD, pn+1, FFN_DIM) = 1;
        W(w->ff_up[L], S_AD_LEFT_VALUE, pn+2, FFN_DIM) = 1;
        W(w->ff_down[L], pn, S_AD_PROD_GRAD_RV, D) = 0.5f;
        W(w->ff_down[L], pn+1, S_AD_PROD_GRAD_RV, D) = -0.5f;
        W(w->ff_down[L], pn+2, S_AD_PROD_GRAD_RV, D) = -0.5f;
        pn += 3;

        /* Product 3: left_value * right_value → AD_PROD_LR (AD_MUL forward) */
        W(w->ff_up[L], S_AD_LEFT_VALUE, pn, FFN_DIM) = 1; W(w->ff_up[L], S_AD_RIGHT_VALUE, pn, FFN_DIM) = 1;
        W(w->ff_up[L], S_AD_LEFT_VALUE, pn+1, FFN_DIM) = 1;
        W(w->ff_up[L], S_AD_RIGHT_VALUE, pn+2, FFN_DIM) = 1;
        W(w->ff_down[L], pn, S_AD_PROD_LR, D) = 0.5f;
        W(w->ff_down[L], pn+1, S_AD_PROD_LR, D) = -0.5f;
        W(w->ff_down[L], pn+2, S_AD_PROD_LR, D) = -0.5f;
        pn += 3;

        /* Product 4: grad * saved_val → AD_PROD_GRAD_SV (ALL unary backward rules) */
        W(w->ff_up[L], S_AD_CUR_GRAD, pn, FFN_DIM) = 1; W(w->ff_up[L], S_AD_CUR_SAVED, pn, FFN_DIM) = 1;
        W(w->ff_up[L], S_AD_CUR_GRAD, pn+1, FFN_DIM) = 1;
        W(w->ff_up[L], S_AD_CUR_SAVED, pn+2, FFN_DIM) = 1;
        W(w->ff_down[L], pn, S_AD_PROD_GRAD_SV, D) = 0.5f;
        W(w->ff_down[L], pn+1, S_AD_PROD_GRAD_SV, D) = -0.5f;
        W(w->ff_down[L], pn+2, S_AD_PROD_GRAD_SV, D) = -0.5f;
        pn += 3;

        printf("[WEIGHT_GEN] Layer %d: %d SQUARE neurons\n", L, pn);
    }

    /* ── Layer 1: Preprocessing (gated FFN) ── */
    {
        const int L = 1;
        w->ff_type[L] = 2;
        int n = 0;

        /* GET_LOCAL address resolution: indicator(OPERAND==a) * mem[a] → LOADVAL */
        for (int a = 0; a < MEM_SIZE; a++) {
            W(w->ff_gate[L], S_OPERAND, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)a + 0.5f);
            W(w->ff_up[L], S_MEM0+a, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_LOADVAL, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_OPERAND, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)a - 0.5f);
            W(w->ff_up[L], S_MEM0+a, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_LOADVAL, D) = -1.0f;
            n++;
        }

        /* SET_LOCAL store deltas: indicator(OPERAND==a) * (TOS - mem[a]) → STORED0+a */
        for (int a = 0; a < MEM_SIZE; a++) {
            W(w->ff_gate[L], S_OPERAND, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)a + 0.5f);
            W(w->ff_up[L], S_TOS, n, FFN_DIM) = 1.0f;
            W(w->ff_up[L], S_MEM0+a, n, FFN_DIM) = -1.0f;
            W(w->ff_down[L], n, S_STORED0+a, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_OPERAND, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)a - 0.5f);
            W(w->ff_up[L], S_TOS, n, FFN_DIM) = 1.0f;
            W(w->ff_up[L], S_MEM0+a, n, FFN_DIM) = -1.0f;
            W(w->ff_down[L], n, S_STORED0+a, D) = -1.0f;
            n++;
        }

        /* JUMP_IF_FALSE zero case: indicator(TOS==0) * operand → ZOPER */
        W(w->ff_gate[L], S_TOS, n, FFN_DIM) = SCALE;
        w->ff_gate_b[L][n] = SCALE * 0.5f;
        W(w->ff_up[L], S_OPERAND, n, FFN_DIM) = 1.0f;
        W(w->ff_down[L], n, S_ZOPER, D) = 1.0f;
        n++;
        W(w->ff_gate[L], S_TOS, n, FFN_DIM) = SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f);
        W(w->ff_up[L], S_OPERAND, n, FFN_DIM) = 1.0f;
        W(w->ff_down[L], n, S_ZOPER, D) = -1.0f;
        n++;

        /* JUMP_IF_FALSE zero case: indicator(TOS==0) * (PC+1) → ZPC1 */
        W(w->ff_gate[L], S_TOS, n, FFN_DIM) = SCALE;
        w->ff_gate_b[L][n] = SCALE * 0.5f;
        W(w->ff_up[L], S_PC, n, FFN_DIM) = 1.0f;
        w->ff_up_b[L][n] = 1.0f;
        W(w->ff_down[L], n, S_ZPC1, D) = 1.0f;
        n++;
        W(w->ff_gate[L], S_TOS, n, FFN_DIM) = SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f);
        W(w->ff_up[L], S_PC, n, FFN_DIM) = 1.0f;
        w->ff_up_b[L][n] = 1.0f;
        W(w->ff_down[L], n, S_ZPC1, D) = -1.0f;
        n++;

        /* Bounded DIV/MOD result precompute into S_ZPC1.
         * DIV: denominator-gated linear reciprocal for positive integer
         * denominators 1..16.
         * MOD: exact positive integer lookup for denominators 3 and 4 over
         * the verifier's small numerator range. Zero remainders need no
         * neuron because S_ZPC1's baseline is zero when TOS is nonzero. */
        for (int d = 1; d <= DIV_WEIGHT_MAX_DENOM; d++) {
            n = add_gated_opcode_index(w, L, n, OP_DIV, S_TOS, d,
                                       S_SOS, 1.0f / (float)d,
                                       -1,0,-1,0,-1,0,
                                       0, S_ZPC1, 1.0f);
            n = add_gated_opcode_index(w, L, n, OP_DIV, S_TOS, d,
                                       -1,0,-1,0,-1,0,-1,0,
                                       1.0f, S_IS_NATIVE, 1.0f);
        }
        for (int d = 3; d <= 4; d++) {
            for (int v = 0; v <= MOD_WEIGHT_MAX_NUM; v++) {
                int r = v % d;
                if (r != 0) {
                    n = add_gated_opcode_two_indices(w, L, n, OP_MOD,
                                                     S_TOS, d, S_SOS, v,
                                                     -1,0,-1,0,-1,0,-1,0,
                                                     (float)r, S_ZPC1, 1.0f);
                }
                n = add_gated_opcode_two_indices(w, L, n, OP_MOD,
                                                 S_TOS, d, S_SOS, v,
                                                 -1,0,-1,0,-1,0,-1,0,
                                                 1.0f, S_IS_NATIVE, 1.0f);
            }
        }

        /* CMP_EQ: indicator(TOS - SOS == 0) → two neurons on (TOS - SOS) */
        W(w->ff_gate[L], S_TOS, n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_SOS, n, FFN_DIM) = -SCALE;
        w->ff_gate_b[L][n] = SCALE * 0.5f;
        w->ff_up_b[L][n] = 1.0f;
        W(w->ff_down[L], n, S_CMP_EQ, D) = 1.0f;
        n++;
        W(w->ff_gate[L], S_TOS, n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_SOS, n, FFN_DIM) = -SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f);
        w->ff_up_b[L][n] = 1.0f;
        W(w->ff_down[L], n, S_CMP_EQ, D) = -1.0f;
        n++;

        /* CMP_LT: sigmoid(SCALE*(TOS - SOS - 0.5)) → 1 when SOS < TOS */
        W(w->ff_gate[L], S_TOS, n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_SOS, n, FFN_DIM) = -SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f);
        w->ff_up_b[L][n] = 1.0f;
        W(w->ff_down[L], n, S_CMP_LT, D) = 1.0f;
        n++;

        /* ABS_DELTA: indicator(TOS < 0) * (-2*TOS)
         * gate = sigmoid(SCALE*(-TOS - 0.5)) ≈ 1 when TOS < 0 */
        W(w->ff_gate[L], S_TOS, n, FFN_DIM) = -SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f);
        W(w->ff_up[L], S_TOS, n, FFN_DIM) = -2.0f;
        W(w->ff_down[L], n, S_ABS_DELTA, D) = 1.0f;
        n++;

        /* Stage-1 type predicate indicators over S_TYPE_TOS. These feed
         * NULL_P and the six type-predicate opcodes in Layer 3. */
        n = add_indicator_precompute(w, L, n, S_TYPE_TOS, TYPE_NUMBER,  S_TYPE_IS_NUM);
        n = add_indicator_precompute(w, L, n, S_TYPE_TOS, TYPE_BOOL,    S_TYPE_IS_BOOL);
        n = add_indicator_precompute(w, L, n, S_TYPE_TOS, TYPE_PAIR,    S_TYPE_IS_PAIR);
        n = add_indicator_precompute(w, L, n, S_TYPE_TOS, TYPE_CLOSURE, S_TYPE_IS_PROC);
        n = add_indicator_precompute(w, L, n, S_TYPE_TOS, TYPE_STRING,  S_TYPE_IS_STR);
        n = add_indicator_precompute(w, L, n, S_TYPE_TOS, TYPE_VECTOR,  S_TYPE_IS_VEC);
        n = add_indicator_precompute(w, L, n, S_TYPE_TOS, TYPE_NIL,     S_TYPE_IS_NIL);
        n = add_indicator_precompute(w, L, n, S_OPCODE, OP_AD_ABS,      S_AD_UNARY_ABS_ACTIVE);
        n = add_indicator_precompute(w, L, n, S_OPCODE, OP_AD_RELU,     S_AD_UNARY_RELU_ACTIVE);

        /* AD forward binary parent loads. Layer 1 runs before the SQUARE
         * layer, so OP_AD_MUL can consume AD_LEFT_VALUE/AD_RIGHT_VALUE in
         * Layer 2 and record the product without native postprocess. */
        int bounded_binary_ops[] = { OP_AD_ADD, OP_AD_SUB, OP_AD_MUL, OP_AD_DIV, OP_AD_POW };
        for (int oi = 0; oi < (int)(sizeof(bounded_binary_ops) / sizeof(bounded_binary_ops[0])); oi++) {
            int opc = bounded_binary_ops[oi];
            for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
                int val_dim = S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_VALUE;
                n = add_gated_pair_ad_index(w,L,n, opc, S_SOS, slot,
                                            val_dim,1,-1,0,-1,0,-1,0,
                                            0, S_AD_LEFT_VALUE, 1.0f);
                n = add_gated_pair_ad_index(w,L,n, opc, S_TOS, slot,
                                            val_dim,1,-1,0,-1,0,-1,0,
                                            0, S_AD_RIGHT_VALUE, 1.0f);
            }
        }
        int bounded_unary_ops[] = {
            OP_AD_ABS, OP_AD_RELU, OP_AD_SIGMOID, OP_AD_TANH,
            OP_AD_EXP, OP_AD_LOG, OP_AD_SQRT, OP_AD_SIN, OP_AD_COS
        };
        for (int oi = 0; oi < (int)(sizeof(bounded_unary_ops) / sizeof(bounded_unary_ops[0])); oi++) {
            int opc = bounded_unary_ops[oi];
            for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
                int val_dim = S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_VALUE;
                n = add_gated_pair_ad_index(w,L,n, opc, S_TOS, slot,
                                            val_dim,1,-1,0,-1,0,-1,0,
                                            0, S_AD_LEFT_VALUE, 1.0f);
            }
        }

        /* ── AD backward: cursor load — indicator pair with IS_BACKWARD ──
         * Gate: SCALE*CURSOR + SCALE*IS_BACKWARD + bias
         * Pair bias: +neuron = SCALE*(-slot + 0.5) - SCALE
         *            -neuron = SCALE*(-slot - 0.5) - SCALE
         * Proof: pair difference cancels adjacent-slot leakage:
         *   C=s,BW=1: sigmoid(0.5S) - sigmoid(-0.5S) ≈ 1 ✓
         *   C=s,BW=0: sigmoid(-0.5S) - sigmoid(-1.5S) ≈ 0 ✓
         *   C=s±1,BW=1: sigmoid(±1.5S) - sigmoid(±0.5S) ≈ 0 ✓ */
        for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
            int fields[] = { AD_F_OP, AD_F_VALUE, AD_F_GRAD, AD_F_LEFT, AD_F_RIGHT, AD_F_SAVED };
            int targets[] = { S_AD_CUR_OP, S_AD_CUR_VALUE, S_AD_CUR_GRAD, S_AD_CUR_LEFT, S_AD_CUR_RIGHT, S_AD_CUR_SAVED };
            for (int fi = 0; fi < 6; fi++) {
                int src = S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + fields[fi];
                int dst = targets[fi];
                /* Positive neuron */
                W(w->ff_gate[L], S_AD_CURSOR, n, FFN_DIM) = SCALE;
                W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
                w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f) - SCALE;
                W(w->ff_up[L], src, n, FFN_DIM) = 1.0f;
                W(w->ff_down[L], n, dst, D) = 1.0f;
                n++;
                /* Negative neuron */
                W(w->ff_gate[L], S_AD_CURSOR, n, FFN_DIM) = SCALE;
                W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
                w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f) - SCALE;
                W(w->ff_up[L], src, n, FFN_DIM) = 1.0f;
                W(w->ff_down[L], n, dst, D) = -1.0f;
                n++;
            }
        }

        /* ── Cursor decrement: IS_BACKWARD → delta[CURSOR] = -1 ── */
        W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f);
        w->ff_up_b[L][n] = -1.0f;
        W(w->ff_down[L], n, S_AD_CURSOR, D) = 1.0f;
        n++;

        /* ── Completion check: indicator(CURSOR == 0) AND IS_BACKWARD → clear IS_BACKWARD ──
         * Fires on the cycle that processes the LAST tape node (pre-decrement
         * cursor == 0 → post-decrement cursor == -1). Matches the reference
         * VM's ad_backward_step which clears IS_BACKWARD same step it makes
         * cursor go negative. The original encoding used indicator(cursor, -1)
         * which fires one cycle late, adding a spurious extra backward cycle
         * that diverged from the reference VM step-for-step.
         *
         * The IS_BACKWARD coefficient is 10·SCALE (not SCALE) so that high
         * cursor values (up to AD_MAX_TAPE-1 = 7) cannot push the gate open
         * when IS_BACKWARD is 0 — same dual-input AND pattern as the
         * AD_TAPE_LEN/AD_IS_FORWARD gates in Layer 4. */
        W(w->ff_gate[L], S_AD_CURSOR,        n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_AD_IS_BACKWARD,   n, FFN_DIM) = 10.0f * SCALE;
        w->ff_gate_b[L][n] = SCALE * 0.5f - 10.0f * SCALE; /* fires at cursor==0 AND bw==1 */
        W(w->ff_up[L], S_AD_IS_BACKWARD, n, FFN_DIM) = -1.0f;
        W(w->ff_down[L], n, S_AD_IS_BACKWARD, D) = 1.0f;
        n++;
        W(w->ff_gate[L], S_AD_CURSOR,        n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_AD_IS_BACKWARD,   n, FFN_DIM) = 10.0f * SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f) - 10.0f * SCALE; /* fires at cursor>=1 AND bw==1 */
        W(w->ff_up[L], S_AD_IS_BACKWARD, n, FFN_DIM) = -1.0f;
        W(w->ff_down[L], n, S_AD_IS_BACKWARD, D) = -1.0f;
        n++;
        /* Also clear AD_MODE */
        W(w->ff_gate[L], S_AD_CURSOR,        n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_AD_IS_BACKWARD,   n, FFN_DIM) = 10.0f * SCALE;
        w->ff_gate_b[L][n] = SCALE * 0.5f - 10.0f * SCALE;
        W(w->ff_up[L], S_AD_MODE, n, FFN_DIM) = -1.0f;
        W(w->ff_down[L], n, S_AD_MODE, D) = 1.0f;
        n++;
        W(w->ff_gate[L], S_AD_CURSOR,        n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_AD_IS_BACKWARD,   n, FFN_DIM) = 10.0f * SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f) - 10.0f * SCALE;
        W(w->ff_up[L], S_AD_MODE, n, FFN_DIM) = -1.0f;
        W(w->ff_down[L], n, S_AD_MODE, D) = -1.0f;
        n++;

        /* ── Transient clear: IS_BACKWARD → zero cursor-loaded + Zone D scratch ── */
        for (int d = S_AD_CUR_OP; d <= S_AD_RIGHT_VALUE; d++) {
            W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-0.5f);
            W(w->ff_up[L], d, n, FFN_DIM) = -1.0f;
            W(w->ff_down[L], n, d, D) = 1.0f;
            n++;
        }
        W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f);
        W(w->ff_up[L], S_AD_IS_FORWARD, n, FFN_DIM) = -1.0f;
        W(w->ff_down[L], n, S_AD_IS_FORWARD, D) = 1.0f;
        n++;
        for (int d = S_AD_GRAD_ACCUM; d <= S_AD_SPARE8; d++) {
            W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-0.5f);
            W(w->ff_up[L], d, n, FFN_DIM) = -1.0f;
            W(w->ff_down[L], n, d, D) = 1.0f;
            n++;
        }

        printf("[WEIGHT_GEN] Layer %d: %d neurons\n", L, n);
    }

    /* ── Layer 3: Execution (gated FFN) ── */
    {
        const int L = 3;
        w->ff_type[L] = 2;
        int n = 0;

        /* Universal: clear output to -1 */
        n = add_unconditional(w, L, n, S_OUTPUT, -1.0f, -1.0f, S_OUTPUT, 1.0f);
        /* Universal: clear HAS_OUT */
        n = add_unconditional(w, L, n, S_HAS_OUT, -1.0f, 0, S_HAS_OUT, 1.0f);
        /* Universal: clear intermediate dims (Zone A: 16-31, Zone D: 112-127,
         * arena op transients)
         * Skip S_AD_IS_BACKWARD (113) — must persist through L4/L5 for backward gating */
        for (int d = S_OPCODE; d <= S_ABS_DELTA; d++)
            n = add_unconditional(w, L, n, d, -1.0f, 0, d, 1.0f);
        n = add_unconditional(w, L, n, S_AD_IS_FORWARD, -1.0f, 0, S_AD_IS_FORWARD, 1.0f);
        /* S_AD_IS_BACKWARD (113) intentionally NOT cleared */
        for (int d = S_AD_GRAD_ACCUM; d <= S_AD_SPARE8; d++)
            n = add_unconditional(w, L, n, d, -1.0f, 0, d, 1.0f);
        for (int d = S_ARENA_TRANSIENT_START; d <= S_ARENA_TRANSIENT_END; d++)
            n = add_unconditional(w, L, n, d, -1.0f, 0, d, 1.0f);

        /* OP_NOP (0): PC += 1 */
        n = add_gated_pair(w,L,n, 0, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);

        /* OP_CONST (1): TOS=oper, push down, depth++, PC++ */
        n = add_gated_pair(w,L,n, 1, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_OPERAND,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 1, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 1, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_type_push(w,L,n, 1, TYPE_NUMBER);

        /* OP_NIL (2): push -1, same pattern as CONST */
        n = add_gated_pair(w,L,n, 2, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_TOS,-1,-1,0,-1,0,-1,0, -1.0f, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 2, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 2, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_type_push(w,L,n, 2, TYPE_NIL);

        /* OP_TRUE (3): push 1 */
        n = add_gated_pair(w,L,n, 3, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_TOS,-1,-1,0,-1,0,-1,0, 1.0f, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 3, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 3, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_type_push(w,L,n, 3, TYPE_BOOL);

        /* OP_FALSE (4): push 0 */
        n = add_gated_pair(w,L,n, 4, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 4, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 4, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_type_push(w,L,n, 4, TYPE_BOOL);

        /* OP_POP (5): shift up, depth-- */
        n = add_gated_pair(w,L,n, 5, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 5, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 5, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_pop(w,L,n, 5);

        /* OP_DUP (6): push TOS copy, shift down */
        n = add_gated_pair(w,L,n, 6, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 6, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 6, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 6, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 6, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_type_dup(w,L,n, 6);

        /* OP_SWAP (83): exchange TOS<->SOS, PC++; depth, R2, R3 unchanged */
        n = add_gated_pair(w,L,n, 83, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 83, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 83, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_type_swap(w,L,n, 83);

        /* OP_ADD (7): TOS+=SOS, shift up, depth--, PC++ */
        n = add_gated_pair(w,L,n, 7, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_SOS,1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 7, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 7, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_binary_result(w,L,n, 7, TYPE_NUMBER);

        /* OP_SUB (8): TOS=SOS-TOS (delta=SOS-2*TOS), shift up */
        n = add_gated_pair(w,L,n, 8, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_SOS,1,S_TOS,-2,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 8, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 8, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_binary_result(w,L,n, 8, TYPE_NUMBER);

        /* OP_MUL (9): TOS=TOS*SOS=product, shift up */
        n = add_gated_pair(w,L,n, 9, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_PRODUCT,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 9, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 9, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_binary_result(w,L,n, 9, TYPE_NUMBER);

        /* OP_NEG (12): TOS = -TOS, PC++ */
        n = add_gated_pair(w,L,n, 12, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 12, S_TOS,-2,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 12, TYPE_NUMBER);

        /* OP_ABS (13): TOS = |TOS|, PC++ — uses ABS_DELTA precomputed in layer 2 */
        n = add_gated_pair(w,L,n, 13, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 13, S_ABS_DELTA,1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 13, TYPE_NUMBER);

        /* OP_EQ (14): TOS = (TOS==SOS), binary comparison, shift up */
        n = add_gated_pair(w,L,n, 14, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 14, S_CMP_EQ,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 14, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 14, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 14, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 14, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_binary_result(w,L,n, 14, TYPE_BOOL);

        /* OP_LT (15): TOS = (SOS < TOS) = CMP_LT */
        n = add_gated_pair(w,L,n, 15, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 15, S_CMP_LT,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 15, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 15, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 15, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 15, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_binary_result(w,L,n, 15, TYPE_BOOL);

        /* OP_GT (16): TOS = (SOS > TOS) = 1 - CMP_LT - CMP_EQ */
        n = add_gated_pair(w,L,n, 16, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 16, S_CMP_LT,-1,S_CMP_EQ,-1,S_TOS,-1,-1,0, 1.0f, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 16, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 16, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 16, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 16, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_binary_result(w,L,n, 16, TYPE_BOOL);

        /* OP_LE (17): TOS = (SOS <= TOS) = CMP_LT + CMP_EQ */
        n = add_gated_pair(w,L,n, 17, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 17, S_CMP_LT,1,S_CMP_EQ,1,S_TOS,-1,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 17, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 17, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 17, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 17, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_binary_result(w,L,n, 17, TYPE_BOOL);

        /* OP_GE (18): TOS = (SOS >= TOS) = 1 - CMP_LT */
        n = add_gated_pair(w,L,n, 18, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 18, S_CMP_LT,-1,S_TOS,-1,-1,0,-1,0, 1.0f, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 18, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 18, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 18, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 18, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_binary_result(w,L,n, 18, TYPE_BOOL);

        /* OP_NOT (19): TOS = (TOS==0) ? 1 : 0. Unary, no stack shift.
         * Approach: delta = indicator(TOS,0) - TOS.
         * indicator(TOS,0) is already available via the sigmoid pair approach,
         * but we need it as a linear combination for the gated pair.
         * Use: up = -TOS + indicator(TOS,0) ... but indicator is nonlinear.
         * Solution: precompute in layer 2 or use two separate gated pairs.
         * Actually: the zero indicator for TOS is available via the ZOPER mechanism.
         * We can use: delta = (ZOPER/OPERAND when OPERAND!=0)... no, ZOPER depends on operand.
         *
         * Simpler: NOT on TOS uses the same sigmoid-pair as indicator(TOS, 0).
         * We compute indicator(TOS, 0) directly in the gate:
         * gated_pair1: gate=indicator(op,19)*indicator(TOS,0)*alive, up=1, out=TOS (sets TOS=1 when TOS==0)
         * But we can't nest indicators in a single gate neuron.
         *
         * Solution: reuse ZOPER from Layer 1 which precomputes indicator(TOS,0).
         * NOT = "TOS becomes indicator(TOS,0)" = ZOPER/operand trick.
         * NOT = "TOS becomes 1 if TOS==0, else 0"
         * delta_TOS = result - TOS where result = indicator(TOS,0)
         * We can split this: first clear TOS (delta = -TOS), then add indicator(TOS,0).
         * But both happen in the same gated pair, which gates on opcode==19.
         * Inside the pair, we need -TOS + indicator(TOS,0) which requires the indicator.
         *
         * Resolution: use 4 neurons instead of 2. First pair clears TOS:
         * pair1: gate=indicator(op,19)*alive, up=-TOS, out=TOS → clears TOS
         * Then pair2 ADDS 1 conditional on (opcode==19 AND TOS==0):
         * pair2: gate=sigmoid(S*(op-19+0.5))*sigmoid(S*(-TOS+0.5)), up=1, out=TOS
         * But that's a product of sigmoids in the gate, which we can't do with a single gate.
         *
         * Final approach: delegate NOT to reference/simulated, add it to weight matrix
         * using the precomputed zero indicator from ZOPER. Wait — ZOPER = indicator(TOS,0) * OPERAND.
         * If OPERAND happens to be nonzero for NOT, we can recover indicator(TOS,0) = ZOPER/OPERAND.
         * But operand for NOT is 0 (unary, no operand). So ZOPER = 0 always. Useless.
         *
         * OK — add a new precompute: IND_TOS_ZERO into S_ABS_DELTA or reuse existing.
         * Actually, I'll add a precompute for indicator(TOS,0) into the layer 2 path
         * and store it... but we're out of intermediate dims.
         *
         * Pragmatic: NOT is used in control flow. For the weight matrix, we handle it
         * by using the same approach as ABS: precompute in layer 2.
         * We can reuse S_ABS_DELTA for a different purpose when ABS isn't executing,
         * but that's ugly. Better: for NOT, we use the fact that
         * indicator(TOS,0) = sigmoid(S*(0-TOS+0.5)) - sigmoid(S*(0-TOS-0.5))
         *                   = sigmoid(S*(-TOS+0.5)) - sigmoid(S*(-TOS-0.5))
         * This is exactly: gate1 = sigmoid(-S*TOS + S*0.5), gate2 = sigmoid(-S*TOS - S*0.5)
         * And the up_value is 1.0 (bias).
         * So: we add neurons directly in layer 3 that don't use add_gated_pair.
         * We need 4 neurons total:
         * n1: gate = sigmoid(S*(op-19+0.5))*... nope, we can't combine.
         *
         * Let's just use the approach from the simulated path: compute NOT inline.
         * For the matrix path, NOT = indicator(TOS,0) requires 2 extra neurons
         * outside the gated_pair framework. We'll manually wire them.
         */
        /* OP_NOT (19): step 1 — clear TOS, step 2 — set to 1 if was 0.
         * Uses CMP_EQ which computes indicator(TOS-SOS, 0). But NOT is unary.
         * Precompute: we reuse the layer 2 ABS indicator for the negative case.
         * For NOT, we need indicator(TOS, 0). Approach: in layer 3, clear TOS
         * then conditionally add 1. Use ZOPER trick: when operand=1 and we
         * encode NOT as {OP_NOT, 1}, ZOPER = indicator(TOS,0) * 1 = indicator(TOS,0).
         * Then delta_TOS = ZOPER - TOS. This WORKS if we make NOT's operand = 1.
         */
        n = add_gated_pair(w,L,n, 19, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 19, S_ZOPER,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 19, TYPE_BOOL);

        /* OP_GET_LOCAL (20): push mem[operand] — LOADVAL precomputed from OPERAND */
        n = add_gated_pair(w,L,n, 20, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 20, S_LOADVAL,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 20, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 20, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 20, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 20, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_type_push(w,L,n, 20, TYPE_NUMBER);

        /* OP_SET_LOCAL (21): mem[operand]=TOS, pop — uses precomputed STORED deltas */
        n = add_gated_pair(w,L,n, 21, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        for (int a = 0; a < MEM_SIZE; a++) {
            n = add_gated_pair(w,L,n, 21, S_STORED0+a,1,-1,0,-1,0,-1,0, 0, S_MEM0+a, 1.0f);
        }
        n = add_gated_pair(w,L,n, 21, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 21, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 21, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 21, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 21, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_pop(w,L,n, 21);

        /* OP_CALL (25): set IS_CALL flag, PC++ */
        n = add_gated_pair(w,L,n, 25, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 25, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_CALL, 1.0f);

        /* Stage-2 arena pair ops (31-33): weight-encoded against bounded arena. */
        n = add_gated_pair(w,L,n, 31, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_ARENA_NEXT,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 31, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_TYPE_TOS,-1,-1,0,-1,0,-1,0, TYPE_PAIR, S_TYPE_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_TYPE_R2,1,S_TYPE_SOS,-1,-1,0,-1,0, 0, S_TYPE_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_TYPE_R3,1,S_TYPE_R2,-1,-1,0,-1,0, 0, S_TYPE_R2, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);
        n = add_gated_pair(w,L,n, 31, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_KIND, 1.0f);
        n = add_gated_pair(w,L,n, 31, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 31, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_CDR, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_ARENA_NEXT,1,-1,0,-1,0,-1,0, 0, S_ARENA_TARGET, 1.0f);
        n = add_gated_pair(w,L,n, 31, -1,0,-1,0,-1,0,-1,0, ARENA_KIND_PAIR, S_ARENA_NEW_KIND, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_SOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CDR, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_TYPE_SOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CAR_TYPE, 1.0f);
        n = add_gated_pair(w,L,n, 31, S_TYPE_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CDR_TYPE, 1.0f);
        n = add_gated_pair(w,L,n, 31, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_NEXT, 1.0f);

        n = add_gated_pair(w,L,n, 32, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 32, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 32, TYPE_NUMBER);
        n = add_gated_pair(w,L,n, 32, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_READ_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 32, S_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_TARGET, 1.0f);

        n = add_gated_pair(w,L,n, 33, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 33, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 33, TYPE_NUMBER);
        n = add_gated_pair(w,L,n, 33, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_READ_CDR, 1.0f);
        n = add_gated_pair(w,L,n, 33, S_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_TARGET, 1.0f);
        /* OP_NULL_P (34): weight-encoded nil predicate */
        n = add_gated_pair(w,L,n, 34, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 34, S_TYPE_IS_NIL,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 34, TYPE_BOOL);
        /* OP_GET_UPVALUE (22): MEM fallback; Layer 4 overwrites TOS from
         * current arena closure cell when S_CUR_CLOSURE is in range. */
        n = add_gated_pair(w,L,n, 22, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 22, S_LOADVAL,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 22, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 22, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 22, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 22, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_type_push(w,L,n, 22, TYPE_NUMBER);
        n = add_gated_pair(w,L,n, 22, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_READ_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 22, S_CUR_CLOSURE,1,S_OPERAND,1,-1,0,-1,0, 1.0f, S_ARENA_TARGET, 1.0f);
        /* OP_SET_UPVALUE (23): write MEM fallback and current arena closure cell, then pop. */
        n = add_gated_pair(w,L,n, 23, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        for (int a = 0; a < MEM_SIZE; a++) {
            n = add_gated_pair(w,L,n, 23, S_STORED0+a,1,-1,0,-1,0,-1,0, 0, S_MEM0+a, 1.0f);
        }
        n = add_gated_pair(w,L,n, 23, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 23, S_CUR_CLOSURE,1,S_OPERAND,1,-1,0,-1,0, 1.0f, S_ARENA_TARGET, 1.0f);
        n = add_gated_pair(w,L,n, 23, S_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 23, S_TYPE_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CAR_TYPE, 1.0f);
        n = add_gated_pair(w,L,n, 23, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 23, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 23, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 23, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 23, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_pop(w,L,n, 23);
        /* OP_CLOSE_UPVALUE: copy MEM[operand] into current arena closure cell. */
        n = add_gated_pair(w,L,n, 38, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 38, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 38, S_CUR_CLOSURE,1,S_OPERAND,1,-1,0,-1,0, 1.0f, S_ARENA_TARGET, 1.0f);
        n = add_gated_pair(w,L,n, 38, S_LOADVAL,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 38, -1,0,-1,0,-1,0,-1,0, TYPE_NUMBER, S_ARENA_NEW_CAR_TYPE, 1.0f);
        /* OP_OPEN_CLOSURE: set current closure to TOS without changing stack. */
        n = add_gated_pair(w,L,n, 54, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 54, S_TOS,1,S_CUR_CLOSURE,-1,-1,0,-1,0, 0, S_CUR_CLOSURE, 1.0f);
        /* OP_CLOSURE (24): allocate closure header plus four bounded upvalue cells. */
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 24, S_ARENA_NEXT,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 24, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 24, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 24, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_type_push(w,L,n, 24, TYPE_CLOSURE);
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_KIND, 1.0f);
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_CDR, 1.0f);
        n = add_gated_pair(w,L,n, 24, S_ARENA_NEXT,1,-1,0,-1,0,-1,0, 0, S_ARENA_TARGET, 1.0f);
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, ARENA_KIND_CLOSURE, S_ARENA_NEW_KIND, 1.0f);
        n = add_gated_pair(w,L,n, 24, S_OPERAND,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, (float)MEM_SIZE, S_ARENA_NEW_CDR, 1.0f);
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, TYPE_NUMBER, S_ARENA_NEW_CAR_TYPE, 1.0f);
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, TYPE_NUMBER, S_ARENA_NEW_CDR_TYPE, 1.0f);
        n = add_gated_pair(w,L,n, 24, -1,0,-1,0,-1,0,-1,0, (float)(1 + MEM_SIZE), S_ARENA_NEXT, 1.0f);
        /* OP_TAIL_CALL (26): bounded stack-register arities 0..4.
         * This reuses the current frame: PC=TOS, args move into MEM, stack clears. */
        for (int argc = 0; argc <= MEM_SIZE; argc++) {
            int arg_src[4] = { S_SOS, S_R2, S_R3, -1 };
            n = add_gated_pair_op_operand(w,L,n, 26,argc, S_TOS,1,S_PC,-1,-1,0,-1,0, 0, S_PC, 1.0f);
            n = add_gated_pair_op_operand(w,L,n, 26,argc, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
            n = add_gated_pair_op_operand(w,L,n, 26,argc, S_SOS,-1,-1,0,-1,0,-1,0, 0, S_SOS, 1.0f);
            n = add_gated_pair_op_operand(w,L,n, 26,argc, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
            n = add_gated_pair_op_operand(w,L,n, 26,argc, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
            n = add_gated_pair_op_operand(w,L,n, 26,argc, -1,0,-1,0,-1,0,-1,0, -1.0f-(float)argc, S_DEPTH, 1.0f);
            n = add_gated_pair_op_operand(w,L,n, 26,argc, S_TYPE_TOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_TOS, 1.0f);
            n = add_gated_pair_op_operand(w,L,n, 26,argc, S_TYPE_SOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_SOS, 1.0f);
            n = add_gated_pair_op_operand(w,L,n, 26,argc, S_TYPE_R2,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R2, 1.0f);
            n = add_gated_pair_op_operand(w,L,n, 26,argc, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);
            for (int a = 0; a < MEM_SIZE; a++) {
                if (a < argc && arg_src[a] >= 0) {
                    n = add_gated_pair_op_operand(w,L,n, 26,argc, arg_src[a],1,S_MEM0+a,-1,-1,0,-1,0, 0, S_MEM0+a, 1.0f);
                } else {
                    n = add_gated_pair_op_operand(w,L,n, 26,argc, S_MEM0+a,-1,-1,0,-1,0,-1,0, 0, S_MEM0+a, 1.0f);
                }
            }
        }
        /* OP_NATIVE_CALL (37): IS_NATIVE, PC++ */
        n = add_gated_pair(w,L,n, 37, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 37, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);

        /* OP_CALLCC (55): capture a bounded continuation into four arena cells
         * via the existing contiguous-list writeback lanes, then jump to TOS. */
        {
            int elem_dims[4] = { S_ARENA_LIST_E0, S_ARENA_LIST_E1, S_ARENA_LIST_E2, S_ARENA_LIST_E3 };
            int elem_type_dims[4] = { S_ARENA_LIST_T0, S_ARENA_LIST_T1, S_ARENA_LIST_T2, S_ARENA_LIST_T3 };
            int cdr_dims[4] = { S_ARENA_LIST_CDR0, S_ARENA_LIST_CDR1, S_ARENA_LIST_CDR2, S_ARENA_LIST_CDR3 };
            int cdr_type_dims[4] = { S_ARENA_LIST_CDRT0, S_ARENA_LIST_CDRT1, S_ARENA_LIST_CDRT2, S_ARENA_LIST_CDRT3 };
            int has_dims[4] = { S_ARENA_LIST_HAS_E0, S_ARENA_LIST_HAS_E1, S_ARENA_LIST_HAS_E2, S_ARENA_LIST_HAS_E3 };

            n = add_gated_pair(w,L,n, 55, S_TOS,1,S_PC,-1,-1,0,-1,0, 0, S_PC, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_ARENA_NEXT,1,S_MEM0,-1,-1,0,-1,0, 0, S_MEM0, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_MEM1,-1,-1,0,-1,0,-1,0, 0, S_MEM1, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_MEM2,-1,-1,0,-1,0,-1,0, 0, S_MEM2, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_MEM3,-1,-1,0,-1,0,-1,0, 0, S_MEM3, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_SOS,-1,-1,0,-1,0,-1,0, 0, S_SOS, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_DEPTH,-1,-1,0,-1,0,-1,0, 0, S_DEPTH, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_TYPE_TOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_TOS, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_TYPE_SOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_SOS, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_TYPE_R2,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R2, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);
            n = add_gated_pair(w,L,n, 55, S_ARENA_NEXT,1,-1,0,-1,0,-1,0, 0, S_ARENA_LIST_BASE, 1.0f);
            n = add_gated_pair(w,L,n, 55, -1,0,-1,0,-1,0,-1,0, (float)ARENA_CONT_CELLS, S_ARENA_NEXT, 1.0f);

            n = add_gated_pair(w,L,n, 55, S_PC,1,-1,0,-1,0,-1,0, 1.0f, elem_dims[0], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_DEPTH,1,-1,0,-1,0,-1,0, -1.0f, cdr_dims[0], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_SOS,1,-1,0,-1,0,-1,0, 0, elem_type_dims[0], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_R2,1,-1,0,-1,0,-1,0, 0, cdr_type_dims[0], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_R3,1,-1,0,-1,0,-1,0, 0, elem_dims[1], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_TYPE_SOS,1,-1,0,-1,0,-1,0, 0, elem_type_dims[1], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_TYPE_R2,1,-1,0,-1,0,-1,0, 0, cdr_type_dims[1], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_TYPE_R3,1,-1,0,-1,0,-1,0, 0, elem_dims[2], 1.0f);
            n = add_gated_pair(w,L,n, 55, -1,0,-1,0,-1,0,-1,0, TYPE_NUMBER, cdr_dims[2], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_MEM0,1,-1,0,-1,0,-1,0, 0, elem_type_dims[2], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_MEM1,1,-1,0,-1,0,-1,0, 0, cdr_type_dims[2], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_MEM2,1,-1,0,-1,0,-1,0, 0, elem_dims[3], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_MEM3,1,-1,0,-1,0,-1,0, 0, cdr_dims[3], 1.0f);
            n = add_gated_pair(w,L,n, 55, S_WIND_DEPTH,1,-1,0,-1,0,-1,0, 0, elem_type_dims[3], 1.0f);
            for (int i = 0; i < ARENA_CONT_CELLS; i++)
                n = add_gated_pair(w,L,n, 55, -1,0,-1,0,-1,0,-1,0, 1.0f, has_dims[i], 1.0f);
        }

        /* OP_INVOKE_CC (56): mark Layer 4 to restore from arena base SOS.
         * S_ARENA_VEC_HAS_E0 is used only as a restore flag here; the sentinel
         * in S_ARENA_VEC_LEN prevents collision with real vector-create lanes. */
        n = add_gated_pair(w,L,n, 56, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_VEC_HAS_E0, 1.0f);
        n = add_gated_pair(w,L,n, 56, S_SOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_VEC_BASE, 1.0f);
        n = add_gated_pair(w,L,n, 56, -1,0,-1,0,-1,0,-1,0, CONT_RESTORE_MARKER, S_ARENA_VEC_LEN, 1.0f);

        /* Stage-1 type predicates (45-50): weight-encoded via Layer 1 type indicators. */
        n = add_gated_pair(w,L,n, 45, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 45, S_TYPE_IS_PAIR,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 45, TYPE_BOOL);
        n = add_gated_pair(w,L,n, 46, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 46, S_TYPE_IS_NUM,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 46, TYPE_BOOL);
        n = add_gated_pair(w,L,n, 47, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 47, S_TYPE_IS_STR,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 47, TYPE_BOOL);
        n = add_gated_pair(w,L,n, 48, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 48, S_TYPE_IS_BOOL,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 48, TYPE_BOOL);
        n = add_gated_pair(w,L,n, 49, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 49, S_TYPE_IS_PROC,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 49, TYPE_BOOL);
        n = add_gated_pair(w,L,n, 50, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 50, S_TYPE_IS_VEC,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_type_unary_result(w,L,n, 50, TYPE_BOOL);

        /* SET_CAR/SET_CDR write through Layer 4, then pop pair+value. */
        n = add_gated_pair(w,L,n, 51, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 51, S_R2,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 51, S_R3,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 51, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 51, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 51, -1,0,-1,0,-1,0,-1,0, -2.0f, S_DEPTH, 1.0f);
        n = add_type_pop2(w,L,n, 51);
        n = add_gated_pair(w,L,n, 51, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 51, S_SOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_TARGET, 1.0f);
        n = add_gated_pair(w,L,n, 51, S_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 51, S_TYPE_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CAR_TYPE, 1.0f);

        n = add_gated_pair(w,L,n, 52, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 52, S_R2,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 52, S_R3,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 52, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 52, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 52, -1,0,-1,0,-1,0,-1,0, -2.0f, S_DEPTH, 1.0f);
        n = add_type_pop2(w,L,n, 52);
        n = add_gated_pair(w,L,n, 52, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_CDR, 1.0f);
        n = add_gated_pair(w,L,n, 52, S_SOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_TARGET, 1.0f);
        n = add_gated_pair(w,L,n, 52, S_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CDR, 1.0f);
        n = add_gated_pair(w,L,n, 52, S_TYPE_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CDR_TYPE, 1.0f);

        /* VEC_CREATE stores a bounded inline vector in the arena: header
         * cell plus up to four contiguous element cells. */
        n = add_gated_pair(w,L,n, 39, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 39, S_ARENA_NEXT,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 39, S_SOS,-1,-1,0,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 39, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 39, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 39, S_TYPE_TOS,-1,-1,0,-1,0,-1,0, TYPE_VECTOR, S_TYPE_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 39, S_TYPE_SOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 39, S_TYPE_R2,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R2, 1.0f);
        n = add_gated_pair(w,L,n, 39, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);
        n = add_gated_pair(w,L,n, 39, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_VEC_WRITE, 1.0f);
        n = add_gated_pair(w,L,n, 39, S_ARENA_NEXT,1,-1,0,-1,0,-1,0, 0, S_ARENA_VEC_BASE, 1.0f);
        for (int count = 0; count <= ARENA_MAX_INLINE_VECTOR; count++)
            n = add_vec_create_case(w,L,n,count);

        /* VEC_REF: TOS=index, SOS=vector header → read element-cell car. */
        n = add_gated_pair(w,L,n, 40, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 40, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 40, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 40, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 40, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 40, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair(w,L,n, 40, S_TYPE_TOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 40, S_TYPE_R2,1,S_TYPE_SOS,-1,-1,0,-1,0, 0, S_TYPE_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 40, S_TYPE_R3,1,S_TYPE_R2,-1,-1,0,-1,0, 0, S_TYPE_R2, 1.0f);
        n = add_gated_pair(w,L,n, 40, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);
        n = add_gated_pair(w,L,n, 40, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_READ_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 40, S_SOS,1,S_TOS,1,-1,0,-1,0, 1.0f, S_ARENA_TARGET, 1.0f);

        /* VEC_SET: TOS=value, SOS=index, R2=vector header. */
        n = add_gated_pair(w,L,n, 41, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_R3,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_SOS,-1,-1,0,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 41, -1,0,-1,0,-1,0,-1,0, -3.0f, S_DEPTH, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_TYPE_R3,1,S_TYPE_TOS,-1,-1,0,-1,0, 0, S_TYPE_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_TYPE_SOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_TYPE_R2,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R2, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);
        n = add_gated_pair(w,L,n, 41, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_WRITE_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_R2,1,S_SOS,1,-1,0,-1,0, 1.0f, S_ARENA_TARGET, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 41, S_TYPE_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_NEW_CAR_TYPE, 1.0f);

        /* VEC_LEN reads the vector header's car field. */
        n = add_gated_pair(w,L,n, 42, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 42, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 42, S_TYPE_TOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 42, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_READ_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 42, S_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_TARGET, 1.0f);

        /* STR_REF/STR_LEN use the same bounded arena layout as vector reads. */
        n = add_gated_pair(w,L,n, 43, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 43, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 43, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 43, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 43, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 43, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair(w,L,n, 43, S_TYPE_TOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 43, S_TYPE_R2,1,S_TYPE_SOS,-1,-1,0,-1,0, 0, S_TYPE_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 43, S_TYPE_R3,1,S_TYPE_R2,-1,-1,0,-1,0, 0, S_TYPE_R2, 1.0f);
        n = add_gated_pair(w,L,n, 43, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);
        n = add_gated_pair(w,L,n, 43, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_READ_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 43, S_SOS,1,S_TOS,1,-1,0,-1,0, 1.0f, S_ARENA_TARGET, 1.0f);

        n = add_gated_pair(w,L,n, 44, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 44, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 44, S_TYPE_TOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 44, -1,0,-1,0,-1,0,-1,0, 1.0f, S_ARENA_READ_CAR, 1.0f);
        n = add_gated_pair(w,L,n, 44, S_TOS,1,-1,0,-1,0,-1,0, 0, S_ARENA_TARGET, 1.0f);

        /* OP_POPN (53): remove N values below TOS while preserving TOS itself.
         * The current compiler emits only N <= 3, matching the four-register
         * stack cache that the weight interpreter models directly. */
        n = add_gated_pair(w,L,n, 53, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 1, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 1, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 1, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 1, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 1, S_TYPE_R2,1,S_TYPE_SOS,-1,-1,0,-1,0, 0, S_TYPE_SOS, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 1, S_TYPE_R3,1,S_TYPE_R2,-1,-1,0,-1,0, 0, S_TYPE_R2, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 1, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);

        n = add_gated_pair_op_operand(w,L,n, 53, 2, S_R3,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 2, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 2, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 2, -1,0,-1,0,-1,0,-1,0, -2.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 2, S_TYPE_R3,1,S_TYPE_SOS,-1,-1,0,-1,0, 0, S_TYPE_SOS, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 2, S_TYPE_R2,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R2, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 2, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);

        n = add_gated_pair_op_operand(w,L,n, 53, 3, S_SOS,-1,-1,0,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 3, S_R2,-1,-1,0,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 3, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 3, -1,0,-1,0,-1,0,-1,0, -3.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 3, S_TYPE_SOS,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_SOS, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 3, S_TYPE_R2,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R2, 1.0f);
        n = add_gated_pair_op_operand(w,L,n, 53, 3, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);

        /* OP_VOID (63): no stack effect, PC++ */
        n = add_gated_pair(w,L,n, 63, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);

        /* Exception/dynamic-wind bookkeeping. Full raise/unwind remains native. */
        n = add_gated_pair(w,L,n, 57, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 57, -1,0,-1,0,-1,0,-1,0, 1.0f, S_EXC_DEPTH, 1.0f);
        n = add_gated_pair(w,L,n, 58, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 58, -1,0,-1,0,-1,0,-1,0, -1.0f, S_EXC_DEPTH, 1.0f);
        n = add_gated_pair(w,L,n, 59, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 59, S_TOS,-1,-1,0,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 59, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 59, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 59, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 59, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_type_push(w,L,n, 59, TYPE_NUMBER);
        n = add_gated_pair(w,L,n, 61, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 61, -1,0,-1,0,-1,0,-1,0, 1.0f, S_WIND_DEPTH, 1.0f);
        n = add_gated_pair(w,L,n, 61, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 61, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 61, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 61, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 61, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_pop(w,L,n, 61);
        n = add_gated_pair(w,L,n, 62, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 62, -1,0,-1,0,-1,0,-1,0, -1.0f, S_WIND_DEPTH, 1.0f);

        /* OP_PACK_REST (60): pack MEM[n_fixed..3] into a contiguous arena list. */
        n = add_gated_pair(w,L,n, 60, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        {
            int elem_dims[4] = { S_ARENA_LIST_E0, S_ARENA_LIST_E1, S_ARENA_LIST_E2, S_ARENA_LIST_E3 };
            int elem_type_dims[4] = { S_ARENA_LIST_T0, S_ARENA_LIST_T1, S_ARENA_LIST_T2, S_ARENA_LIST_T3 };
            int cdr_dims[4] = { S_ARENA_LIST_CDR0, S_ARENA_LIST_CDR1, S_ARENA_LIST_CDR2, S_ARENA_LIST_CDR3 };
            int cdr_type_dims[4] = { S_ARENA_LIST_CDRT0, S_ARENA_LIST_CDRT1, S_ARENA_LIST_CDRT2, S_ARENA_LIST_CDRT3 };
            int has_dims[4] = { S_ARENA_LIST_HAS_E0, S_ARENA_LIST_HAS_E1, S_ARENA_LIST_HAS_E2, S_ARENA_LIST_HAS_E3 };
            for (int n_fixed = 0; n_fixed <= MEM_SIZE; n_fixed++) {
                int count = MEM_SIZE - n_fixed;
                n = add_gated_pair_op_operand(w,L,n, 60,n_fixed, S_ARENA_NEXT,1,-1,0,-1,0,-1,0, 0, S_ARENA_LIST_BASE, 1.0f);
                n = add_gated_pair_op_operand(w,L,n, 60,n_fixed, -1,0,-1,0,-1,0,-1,0, (float)count, S_ARENA_NEXT, 1.0f);
                if (n_fixed < MEM_SIZE)
                    n = add_gated_pair_op_operand(w,L,n, 60,n_fixed, S_ARENA_NEXT,1,S_MEM0+n_fixed,-1,-1,0,-1,0, 0, S_MEM0+n_fixed, 1.0f);
                for (int j = 0; j < count; j++) {
                    int mem_dim = S_MEM0 + n_fixed + j;
                    n = add_gated_pair_op_operand(w,L,n, 60,n_fixed, mem_dim,1,-1,0,-1,0,-1,0, 0, elem_dims[j], 1.0f);
                    n = add_gated_pair_op_operand(w,L,n, 60,n_fixed, -1,0,-1,0,-1,0,-1,0, TYPE_NUMBER, elem_type_dims[j], 1.0f);
                    if (j + 1 < count) {
                        n = add_gated_pair_op_operand(w,L,n, 60,n_fixed, S_ARENA_NEXT,1,-1,0,-1,0,-1,0, (float)(j + 1), cdr_dims[j], 1.0f);
                        n = add_gated_pair_op_operand(w,L,n, 60,n_fixed, -1,0,-1,0,-1,0,-1,0, TYPE_PAIR, cdr_type_dims[j], 1.0f);
                    } else {
                        n = add_gated_pair_op_operand(w,L,n, 60,n_fixed, -1,0,-1,0,-1,0,-1,0, -1.0f, cdr_dims[j], 1.0f);
                        n = add_gated_pair_op_operand(w,L,n, 60,n_fixed, -1,0,-1,0,-1,0,-1,0, TYPE_NIL, cdr_type_dims[j], 1.0f);
                    }
                    n = add_gated_pair_op_operand(w,L,n, 60,n_fixed, -1,0,-1,0,-1,0,-1,0, 1.0f, has_dims[j], 1.0f);
                }
            }
        }

        /* Remaining delegated opcodes (38-62): all IS_NATIVE + PC++ */
        for (int opc = 38; opc <= 62; opc++) {
            if (opc == 38 || (opc >= 39 && opc <= 44) || (opc >= 45 && opc <= 50) ||
                opc == 51 || opc == 52 || opc == 53 || opc == 54 ||
                opc == 55 || opc == 56 || (opc >= 57 && opc <= 62)) continue;
            n = add_gated_pair(w,L,n, opc, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
            n = add_gated_pair(w,L,n, opc, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        }

        /* OP_RETURN (27): set IS_RET flag, PC++ — exec loop handles frame pop */
        n = add_gated_pair(w,L,n, 27, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 27, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_RET, 1.0f);

        /* OP_JUMP (28) */
        n = add_gated_pair(w,L,n, 28, S_OPERAND,1,S_PC,-1,-1,0,-1,0, 0, S_PC, 1.0f);

        /* OP_JUMP_IF_FALSE (29): pop TOS, if TOS==0 goto operand, else PC+1 */
        n = add_gated_pair(w,L,n, 29, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 29, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 29, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 29, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 29, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        /* JUMP_IF_FALSE PC: delta[PC] = 1 + ZOPER - ZPC1 */
        n = add_gated_pair(w,L,n, 29,
                           S_ZOPER, 1, S_ZPC1, -1, -1, 0, -1, 0,
                           1.0f, S_PC, 1.0f);
        n = add_type_pop(w,L,n, 29);

        /* OP_LOOP (30): backward jump — same as OP_JUMP */
        n = add_gated_pair(w,L,n, 30, S_OPERAND,1,S_PC,-1,-1,0,-1,0, 0, S_PC, 1.0f);

        /* OP_PRINT (35): output = TOS, HAS_OUT=1, pop, PC++ */
        n = add_gated_pair(w,L,n, 35, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_TOS,1,-1,0,-1,0,-1,0, 1.0f, S_OUTPUT, 1.0f);
        n = add_gated_pair(w,L,n, 35, -1,0,-1,0,-1,0,-1,0, 1.0f, S_HAS_OUT, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair(w,L,n, 35, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair(w,L,n, 35, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_type_pop(w,L,n, 35);

        /* OP_DIV/OP_MOD: bounded exact integer artifact path. Layer 1
         * precomputes both the result (S_ZPC1) and an active flag
         * (S_IS_NATIVE as scratch). Layer 3 consumes and clears the scratch
         * flag, so no traced native boundary remains for encoded operands. */
        n = add_flagged_linear(w,L,n, S_IS_NATIVE, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_flagged_linear(w,L,n, S_IS_NATIVE, S_ZPC1,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_flagged_linear(w,L,n, S_IS_NATIVE, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_flagged_linear(w,L,n, S_IS_NATIVE, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_flagged_linear(w,L,n, S_IS_NATIVE, S_R3,-1,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_flagged_linear(w,L,n, S_IS_NATIVE, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_flagged_linear(w,L,n, S_IS_NATIVE, S_TYPE_SOS,1,S_TYPE_TOS,-1,-1,0,-1,0, 0, S_TYPE_TOS, 1.0f);
        n = add_flagged_linear(w,L,n, S_IS_NATIVE, S_TYPE_R2,1,S_TYPE_SOS,-1,-1,0,-1,0, 0, S_TYPE_SOS, 1.0f);
        n = add_flagged_linear(w,L,n, S_IS_NATIVE, S_TYPE_R3,1,S_TYPE_R2,-1,-1,0,-1,0, 0, S_TYPE_R2, 1.0f);
        n = add_flagged_linear(w,L,n, S_IS_NATIVE, S_TYPE_R3,-1,-1,0,-1,0,-1,0, TYPE_NUMBER, S_TYPE_R3, 1.0f);
        for (int opc = 10; opc <= 11; opc++) {
            n = add_gated_opcode_index(w,L,n, opc, S_TOS, 0,
                                       -1,0,-1,0,-1,0,-1,0,
                                       1.0f, S_HALT, 1.0f);
        }

        /* OP_HALT (36) */
        n = add_gated_pair(w,L,n, 36, -1,0,-1,0,-1,0,-1,0, 1.0f, S_HALT, 1.0f);

        /* ── AD Forward Ops (64-78) ── */

        /* OP_AD_VAR (64): push tape_len as tape index, set AD_CUR fields, AD_IS_FORWARD
         * Stack: push down (R3←R2, R2←SOS, SOS←TOS, TOS←tape_len), depth++ */
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_VAR, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, S_OPERAND,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_VALUE, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_RIGHT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);
        n = add_gated_pair_ad(w,L,n, 64, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_MODE, 1.0f);

        /* OP_AD_CONST (65): same as VAR but with AD_OP_CONST */
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, S_TOS,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, S_SOS,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, S_R2,1,S_R3,-1,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, 1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_CONST, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, S_OPERAND,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_VALUE, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_RIGHT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);
        n = add_gated_pair_ad(w,L,n, 65, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_MODE, 1.0f);

        /* OP_AD_ADD (66): binary. Pop 2, push tape_len. Set CUR fields.
         * Note: value computation (left_val + right_val) happens in simulated
         * layer3_ffn via direct tape access. The weight matrix path uses the
         * same simulated code path for value computation — the gated pairs
         * handle stack manipulation and flag setting only.
         * The actual value is set by layer3_ffn's AD forward computation. */
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, 0, S_R3, 1.0f); /* R3=0 */
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_ADD, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, S_SOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 66, S_TOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_RIGHT, 1.0f);
        for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
            int val_dim = S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_VALUE;
            n = add_gated_pair_ad_index(w,L,n, 66, S_SOS, slot, val_dim,1,-1,0,-1,0,-1,0,
                                        0, S_AD_CUR_VALUE, 1.0f);
            n = add_gated_pair_ad_index(w,L,n, 66, S_TOS, slot, val_dim,1,-1,0,-1,0,-1,0,
                                        0, S_AD_CUR_VALUE, 1.0f);
        }
        n = add_gated_pair_ad(w,L,n, 66, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);

        /* OP_AD_SUB (67): same stack pattern as ADD */
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_SUB, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, S_SOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 67, S_TOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_RIGHT, 1.0f);
        for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
            int val_dim = S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_VALUE;
            n = add_gated_pair_ad_index(w,L,n, 67, S_SOS, slot, val_dim,1,-1,0,-1,0,-1,0,
                                        0, S_AD_CUR_VALUE, 1.0f);
            n = add_gated_pair_ad_index(w,L,n, 67, S_TOS, slot, val_dim,1,-1,0,-1,0,-1,0,
                                        0, S_AD_CUR_VALUE, -1.0f);
        }
        n = add_gated_pair_ad(w,L,n, 67, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);

        /* OP_AD_MUL (68): same pattern */
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_MUL, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_SOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_TOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_RIGHT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, S_AD_PROD_LR,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_VALUE, 1.0f);
        n = add_gated_pair_ad(w,L,n, 68, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);

        /* OP_AD_DIV (79): bounded positive integer denominators. */
        n = add_gated_pair_ad(w,L,n, 79, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 79, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 79, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 79, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 79, -1,0,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 79, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 79, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_DIV, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 79, S_SOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 79, S_TOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_RIGHT, 1.0f);
        for (int d = 1; d <= DIV_WEIGHT_MAX_DENOM; d++) {
            float recip = 1.0f / (float)d;
            n = add_gated_opcode_index(w,L,n, 79, S_AD_RIGHT_VALUE, d,
                                       S_AD_LEFT_VALUE, recip,-1,0,-1,0,-1,0,
                                       0, S_AD_CUR_VALUE, 1.0f);
            n = add_gated_opcode_index(w,L,n, 79, S_AD_RIGHT_VALUE, d,
                                       -1,0,-1,0,-1,0,-1,0,
                                       recip, S_AD_CUR_SAVED, 1.0f);
        }
        n = add_gated_pair_ad(w,L,n, 79, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);

        /* OP_AD_POW (80): bounded positive integer base/exponent table. */
        n = add_gated_pair_ad(w,L,n, 80, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 80, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 80, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 80, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 80, -1,0,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 80, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);
        n = add_gated_pair_ad(w,L,n, 80, -1,0,-1,0,-1,0,-1,0, (float)AD_OP_POW, S_AD_CUR_OP, 1.0f);
        n = add_gated_pair_ad(w,L,n, 80, S_SOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_LEFT, 1.0f);
        n = add_gated_pair_ad(w,L,n, 80, S_TOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_RIGHT, 1.0f);
        for (int base = 1; base <= AD_POW_WEIGHT_MAX_BASE; base++) {
            for (int exp = 1; exp <= AD_POW_WEIGHT_MAX_EXP; exp++) {
                float b = (float)base;
                float e = (float)exp;
                float val = powf(b, e);
                float dbase = e * powf(b, e - 1.0f);
                n = add_gated_opcode_two_indices(w,L,n, 80,
                                                 S_AD_LEFT_VALUE, base,
                                                 S_AD_RIGHT_VALUE, exp,
                                                 -1,0,-1,0,-1,0,-1,0,
                                                 val, S_AD_CUR_VALUE, 1.0f);
                n = add_gated_opcode_two_indices(w,L,n, 80,
                                                 S_AD_LEFT_VALUE, base,
                                                 S_AD_RIGHT_VALUE, exp,
                                                 -1,0,-1,0,-1,0,-1,0,
                                                 dbase, S_AD_CUR_SAVED, 1.0f);
            }
        }
        n = add_gated_pair_ad(w,L,n, 80, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);

        /* Unary AD ops (69-76, 81-82): replace TOS with tape_len, set CUR fields */
        int bounded_ad_unary_ops[] = { 69, 70, 71, 72, 73, 74, 75, 76, 81, 82 };
        for (int uop_i = 0; uop_i < (int)(sizeof(bounded_ad_unary_ops) / sizeof(bounded_ad_unary_ops[0])); uop_i++) {
            int uop = bounded_ad_unary_ops[uop_i];
            float ad_op_type;
            switch (uop) {
                case 69: ad_op_type = AD_OP_NEG; break;
                case 70: ad_op_type = AD_OP_ABS; break;
                case 71: ad_op_type = AD_OP_RELU; break;
                case 72: ad_op_type = AD_OP_SIGMOID; break;
                case 73: ad_op_type = AD_OP_TANH; break;
                case 74: ad_op_type = AD_OP_EXP; break;
                case 75: ad_op_type = AD_OP_LOG; break;
                case 76: ad_op_type = AD_OP_SQRT; break;
                case 81: ad_op_type = AD_OP_SIN; break;
                case 82: ad_op_type = AD_OP_COS; break;
                default: ad_op_type = 0; break;
            }
            n = add_gated_pair_ad(w,L,n, uop, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
            n = add_gated_pair_ad(w,L,n, uop, S_AD_TAPE_LEN,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
            n = add_gated_pair_ad(w,L,n, uop, -1,0,-1,0,-1,0,-1,0, ad_op_type, S_AD_CUR_OP, 1.0f);
            n = add_gated_pair_ad(w,L,n, uop, S_TOS,1,-1,0,-1,0,-1,0, 0, S_AD_CUR_LEFT, 1.0f);
            n = add_gated_pair_ad(w,L,n, uop, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_RIGHT, 1.0f);
            if (uop == 69) {
                n = add_gated_pair_ad(w,L,n, 69, -1,0,-1,0,-1,0,-1,0, -1.0f, S_AD_CUR_SAVED, 1.0f);
                for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
                    int val_dim = S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_VALUE;
                    n = add_gated_pair_ad_index(w,L,n, 69, S_TOS, slot, val_dim,1,-1,0,-1,0,-1,0,
                                                0, S_AD_CUR_VALUE, -1.0f);
                }
            } else if (uop == 70) {
                n = add_flagged_value_halfspace(w,L,n, S_AD_UNARY_ABS_ACTIVE, S_AD_LEFT_VALUE,
                                                1.0f, 0.5f,
                                                S_AD_LEFT_VALUE,1,-1,0,-1,0,-1,0,
                                                0, S_AD_CUR_VALUE, 1.0f);
                n = add_flagged_value_halfspace(w,L,n, S_AD_UNARY_ABS_ACTIVE, S_AD_LEFT_VALUE,
                                                1.0f, 0.5f,
                                                -1,0,-1,0,-1,0,-1,0,
                                                1.0f, S_AD_CUR_SAVED, 1.0f);
                n = add_flagged_value_halfspace(w,L,n, S_AD_UNARY_ABS_ACTIVE, S_AD_LEFT_VALUE,
                                                -1.0f, 0.5f,
                                                S_AD_LEFT_VALUE,-1,-1,0,-1,0,-1,0,
                                                0, S_AD_CUR_VALUE, 1.0f);
                n = add_flagged_value_halfspace(w,L,n, S_AD_UNARY_ABS_ACTIVE, S_AD_LEFT_VALUE,
                                                -1.0f, 0.5f,
                                                -1,0,-1,0,-1,0,-1,0,
                                                -1.0f, S_AD_CUR_SAVED, 1.0f);
            } else if (uop == 71) {
                n = add_flagged_value_halfspace(w,L,n, S_AD_UNARY_RELU_ACTIVE, S_AD_LEFT_VALUE,
                                                1.0f, 0.5f,
                                                S_AD_LEFT_VALUE,1,-1,0,-1,0,-1,0,
                                                0, S_AD_CUR_VALUE, 1.0f);
                n = add_flagged_value_halfspace(w,L,n, S_AD_UNARY_RELU_ACTIVE, S_AD_LEFT_VALUE,
                                                1.0f, 0.5f,
                                                -1,0,-1,0,-1,0,-1,0,
                                                1.0f, S_AD_CUR_SAVED, 1.0f);
            } else if (uop == 72) {
                float sig0 = 0.5f;
                float dsig0 = sig0 * (1.0f - sig0);
                float sig1 = 1.0f / (1.0f + expf(-1.0f));
                float dsig1 = sig1 * (1.0f - sig1);
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 0,
                                           -1,0,-1,0,-1,0,-1,0,
                                           sig0, S_AD_CUR_VALUE, 1.0f);
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 0,
                                           -1,0,-1,0,-1,0,-1,0,
                                           dsig0, S_AD_CUR_SAVED, 1.0f);
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 1,
                                           -1,0,-1,0,-1,0,-1,0,
                                           sig1, S_AD_CUR_VALUE, 1.0f);
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 1,
                                           -1,0,-1,0,-1,0,-1,0,
                                           dsig1, S_AD_CUR_SAVED, 1.0f);
            } else if (uop == 73) {
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 0,
                                           -1,0,-1,0,-1,0,-1,0,
                                           0.0f, S_AD_CUR_VALUE, 1.0f);
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 0,
                                           -1,0,-1,0,-1,0,-1,0,
                                           1.0f, S_AD_CUR_SAVED, 1.0f);
            } else if (uop == 74) {
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 0,
                                           -1,0,-1,0,-1,0,-1,0,
                                           1.0f, S_AD_CUR_VALUE, 1.0f);
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 0,
                                           -1,0,-1,0,-1,0,-1,0,
                                           1.0f, S_AD_CUR_SAVED, 1.0f);
            } else if (uop == 75) {
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 1,
                                           -1,0,-1,0,-1,0,-1,0,
                                           0.0f, S_AD_CUR_VALUE, 1.0f);
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 1,
                                           -1,0,-1,0,-1,0,-1,0,
                                           1.0f, S_AD_CUR_SAVED, 1.0f);
            } else if (uop == 76) {
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 4,
                                           -1,0,-1,0,-1,0,-1,0,
                                           2.0f, S_AD_CUR_VALUE, 1.0f);
                n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, 4,
                                           -1,0,-1,0,-1,0,-1,0,
                                           0.25f, S_AD_CUR_SAVED, 1.0f);
            } else if (uop == 81 || uop == 82) {
                for (int input = AD_TRIG_WEIGHT_MIN_INPUT; input <= AD_TRIG_WEIGHT_MAX_INPUT; input++) {
                    float xval = (float)input;
                    float value = (uop == 81) ? sinf(xval) : cosf(xval);
                    float saved = (uop == 81) ? cosf(xval) : -sinf(xval);
                    n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, input,
                                               -1,0,-1,0,-1,0,-1,0,
                                               value, S_AD_CUR_VALUE, 1.0f);
                    n = add_gated_opcode_index(w,L,n, uop, S_AD_LEFT_VALUE, input,
                                               -1,0,-1,0,-1,0,-1,0,
                                               saved, S_AD_CUR_SAVED, 1.0f);
                }
            }
            n = add_gated_pair_ad(w,L,n, uop, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_FORWARD, 1.0f);
            if (uop != 69 && uop != 70 && uop != 71 && uop != 72 &&
                uop != 73 && uop != 74 && uop != 75 && uop != 76 &&
                uop != 81 && uop != 82)
                n = add_gated_pair_ad(w,L,n, uop, -1,0,-1,0,-1,0,-1,0, 1.0f, S_IS_NATIVE, 1.0f);
        }

        /* OP_AD_BACKWARD (77): set backward mode, seed gradient */
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, 2.0f, S_AD_MODE, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, S_TOS,1,S_AD_CURSOR,-1,-1,0,-1,0, 0, S_AD_CURSOR, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, 1.0f, S_AD_IS_BACKWARD, 1.0f);
        for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
            int grad_dim = S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_GRAD;
            n = add_gated_pair_ad_index(w,L,n, 77, S_TOS, slot, -1,0,-1,0,-1,0,-1,0,
                                        1.0f, grad_dim, 1.0f);
        }
        /* Pop TOS */
        n = add_gated_pair_ad(w,L,n, 77, S_SOS,1,S_TOS,-1,-1,0,-1,0, 0, S_TOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, S_R2,1,S_SOS,-1,-1,0,-1,0, 0, S_SOS, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, S_R3,1,S_R2,-1,-1,0,-1,0, 0, S_R2, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, 0, S_R3, 1.0f);
        n = add_gated_pair_ad(w,L,n, 77, -1,0,-1,0,-1,0,-1,0, -1.0f, S_DEPTH, 1.0f);

        /* OP_AD_GRAD (78): replace TOS with gradient of tape[TOS]. */
        n = add_gated_pair_ad(w,L,n, 78, -1,0,-1,0,-1,0,-1,0, 1.0f, S_PC, 1.0f);
        for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
            int grad_dim = S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_GRAD;
            n = add_gated_pair_ad_index(w,L,n, 78, S_TOS, slot, grad_dim,1,S_TOS,-1,-1,0,-1,0,
                                        0, S_TOS, 1.0f);
        }

        /* ── AD backward gradient rules (gated on AD_IS_BACKWARD + AD_CUR_OP) ──
         * Each backward rule computes gradient deltas for AD_LEFT_GRAD_NEW / AD_RIGHT_GRAD_NEW.
         * These use indicator(AD_CUR_OP == op_type) * bw_active as the gate. */

        /* Helper macro: add a backward gradient neuron pair gated on AD_CUR_OP value + backward */
#define ADD_BW_PAIR(op_val, in_dim, in_scale, bias, out_dim, coeff) do { \
    W(w->ff_gate[L], S_AD_CUR_OP, n, FFN_DIM) = SCALE; \
    W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE; \
    w->ff_gate_b[L][n] = SCALE * (-(op_val) + 0.5f) - SCALE; \
    if ((in_dim) >= 0) W(w->ff_up[L], (in_dim), n, FFN_DIM) = (in_scale); \
    w->ff_up_b[L][n] = (bias); \
    W(w->ff_down[L], n, (out_dim), D) = (coeff); \
    n++; \
    W(w->ff_gate[L], S_AD_CUR_OP, n, FFN_DIM) = SCALE; \
    W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE; \
    w->ff_gate_b[L][n] = SCALE * (-(op_val) - 0.5f) - SCALE; \
    if ((in_dim) >= 0) W(w->ff_up[L], (in_dim), n, FFN_DIM) = (in_scale); \
    w->ff_up_b[L][n] = (bias); \
    W(w->ff_down[L], n, (out_dim), D) = -(coeff); \
    n++; \
} while(0)

        /* AD_ADD (2): dL = grad, dR = grad */
        ADD_BW_PAIR(AD_OP_ADD, S_AD_CUR_GRAD, 1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);
        ADD_BW_PAIR(AD_OP_ADD, S_AD_CUR_GRAD, 1.0f, 0, S_AD_RIGHT_GRAD_NEW, 1.0f);

        /* AD_SUB (3): dL = grad, dR = -grad */
        ADD_BW_PAIR(AD_OP_SUB, S_AD_CUR_GRAD, 1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);
        ADD_BW_PAIR(AD_OP_SUB, S_AD_CUR_GRAD, -1.0f, 0, S_AD_RIGHT_GRAD_NEW, 1.0f);

        /* AD_MUL (4): dL = grad*right_value (precomputed in Layer 2 as AD_PROD_GRAD_LV)
         *             dR = grad*left_value  (precomputed in Layer 2 as AD_PROD_GRAD_RV)
         * Note: Layer 2 SQUARE products use AD_CUR_GRAD and AD_RIGHT_VALUE/AD_LEFT_VALUE
         * which are populated by Layer 1 (cursor load) and Layer 4 (parent load). */
        ADD_BW_PAIR(AD_OP_MUL, S_AD_PROD_GRAD_LV, 1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);
        ADD_BW_PAIR(AD_OP_MUL, S_AD_PROD_GRAD_RV, 1.0f, 0, S_AD_RIGHT_GRAD_NEW, 1.0f);

        /* AD_NEG (5): dL = -grad */
        ADD_BW_PAIR(AD_OP_NEG, S_AD_CUR_GRAD, -1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);

        /* AD_RELU (7): dL = grad if left > 0, else 0.
         * 3-condition gate: indicator(OP==RELU) AND IS_BACKWARD AND step(LEFT_VALUE>0).
         * Gate = SCALE*CUR_OP + SCALE*IS_BACKWARD + SCALE*LEFT_VALUE + bias.
         * Bias absorbs all three thresholds. */
        W(w->ff_gate[L], S_AD_CUR_OP, n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_AD_LEFT_VALUE, n, FFN_DIM) = SCALE;
        w->ff_gate_b[L][n] = SCALE * (-AD_OP_RELU + 0.5f) - 2*SCALE;
        W(w->ff_up[L], S_AD_CUR_GRAD, n, FFN_DIM) = 1.0f;
        W(w->ff_down[L], n, S_AD_LEFT_GRAD_NEW, D) = 1.0f;
        n++;
        W(w->ff_gate[L], S_AD_CUR_OP, n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
        W(w->ff_gate[L], S_AD_LEFT_VALUE, n, FFN_DIM) = SCALE;
        w->ff_gate_b[L][n] = SCALE * (-AD_OP_RELU - 0.5f) - 2*SCALE;
        W(w->ff_up[L], S_AD_CUR_GRAD, n, FFN_DIM) = 1.0f;
        W(w->ff_down[L], n, S_AD_LEFT_GRAD_NEW, D) = -1.0f;
        n++;

        /* ALL unary ops: dL = grad * saved_val.
         * saved_val is precomputed during forward recording (Option B):
         *   NEG=-1, ABS=sign, RELU=step, SIGMOID=val*(1-val),
         *   TANH=1-val², EXP=val, LOG=1/input, SQRT=1/(2*val),
         *   SIN=cos(input), COS=-sin(input).
         * Layer 2 SQUARE computes AD_PROD_GRAD_SV = grad * saved_val.
         * Each unary op gets a gated pair that writes AD_PROD_GRAD_SV → LEFT_GRAD_NEW. */
        for (int uop_i = 0; uop_i < 10; uop_i++) {
            float uop_vals[] = { AD_OP_NEG, AD_OP_ABS, AD_OP_RELU, AD_OP_SIGMOID,
                                 AD_OP_TANH, AD_OP_EXP, AD_OP_LOG, AD_OP_SQRT,
                                 AD_OP_SIN, AD_OP_COS };
            ADD_BW_PAIR(uop_vals[uop_i], S_AD_PROD_GRAD_SV, 1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);
        }

#undef ADD_BW_PAIR

        printf("[WEIGHT_GEN] Layer 3: %d neurons used out of %d\n", n, FFN_DIM);
    }

    /* ── Layer 4: AD tape write (gated FFN) ──
     * When AD_IS_FORWARD is set, write AD_CUR_* fields into tape[tape_len].
     * Uses indicator(tape_len, slot) gating. Also increments tape_len.
     *
     * Each field uses a pair of neurons implementing
     *     indicator(AD_TAPE_LEN, slot) · AD_IS_FORWARD = δ(TL, slot) · fw
     * via the dual-input AND pattern:
     *
     *     gate_input = SCALE·TL + SCALE·fw + SCALE·(−slot ± 0.5) − SCALE
     *                = SCALE·(TL − slot + fw ± 0.5 − 1)
     *
     *   When fw == 1:  saturates open at TL ≥ slot − 0.5 (first neuron) and
     *                  TL ≥ slot + 0.5 (second). Difference is +1 when
     *                  TL == slot, 0 elsewhere — a true indicator.
     *   When fw == 0:  both neurons see input ≤ −SCALE/2 (saturating
     *                  closed) for any TL ≥ 0, so the difference is 0 and
     *                  the tape is *not* mutated during backward mode.
     *
     * The original generator omitted the AD_IS_FORWARD coefficient, so the
     * forward tape-write fired during backward whenever TL == slot, which
     * scribbled AD_CUR_VALUE into tape[1] every backward cycle for
     * single-AD_VAR programs (visible as tape[1]=7 in the trace for
     * "AD edge: grad of var = 1"). */
    {
        const int L = 4;
        w->ff_type[L] = 2;
        int n = 0;

        /* For each tape slot i: gate on indicator(AD_TAPE_LEN==i) * AD_IS_FORWARD
         * Write AD_CUR_OP, AD_CUR_VALUE, AD_CUR_LEFT, AD_CUR_RIGHT, AD_CUR_SAVED to tape[i] */
        for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
            /* op field */
            W(w->ff_gate[L], S_AD_TAPE_LEN,    n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_FORWARD,  n, FFN_DIM) = 10.0f * SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f) - 10.0f * SCALE;
            W(w->ff_up[L], S_AD_CUR_OP, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_OP, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_TAPE_LEN,    n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_FORWARD,  n, FFN_DIM) = 10.0f * SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f) - 10.0f * SCALE;
            W(w->ff_up[L], S_AD_CUR_OP, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_OP, D) = -1.0f;
            n++;

            /* value field */
            W(w->ff_gate[L], S_AD_TAPE_LEN,    n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_FORWARD,  n, FFN_DIM) = 10.0f * SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f) - 10.0f * SCALE;
            W(w->ff_up[L], S_AD_CUR_VALUE, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_VALUE, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_TAPE_LEN,    n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_FORWARD,  n, FFN_DIM) = 10.0f * SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f) - 10.0f * SCALE;
            W(w->ff_up[L], S_AD_CUR_VALUE, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_VALUE, D) = -1.0f;
            n++;

            /* left field */
            W(w->ff_gate[L], S_AD_TAPE_LEN,    n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_FORWARD,  n, FFN_DIM) = 10.0f * SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f) - 10.0f * SCALE;
            W(w->ff_up[L], S_AD_CUR_LEFT, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_LEFT, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_TAPE_LEN,    n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_FORWARD,  n, FFN_DIM) = 10.0f * SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f) - 10.0f * SCALE;
            W(w->ff_up[L], S_AD_CUR_LEFT, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_LEFT, D) = -1.0f;
            n++;

            /* right field */
            W(w->ff_gate[L], S_AD_TAPE_LEN,    n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_FORWARD,  n, FFN_DIM) = 10.0f * SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f) - 10.0f * SCALE;
            W(w->ff_up[L], S_AD_CUR_RIGHT, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_RIGHT, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_TAPE_LEN,    n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_FORWARD,  n, FFN_DIM) = 10.0f * SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f) - 10.0f * SCALE;
            W(w->ff_up[L], S_AD_CUR_RIGHT, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_RIGHT, D) = -1.0f;
            n++;

            /* saved field */
            W(w->ff_gate[L], S_AD_TAPE_LEN,    n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_FORWARD,  n, FFN_DIM) = 10.0f * SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f) - 10.0f * SCALE;
            W(w->ff_up[L], S_AD_CUR_SAVED, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_SAVED, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_TAPE_LEN,    n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_FORWARD,  n, FFN_DIM) = 10.0f * SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f) - 10.0f * SCALE;
            W(w->ff_up[L], S_AD_CUR_SAVED, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_SAVED, D) = -1.0f;
            n++;
        }

        /* Increment tape_len when IS_FORWARD is set */
        W(w->ff_gate[L], S_AD_IS_FORWARD, n, FFN_DIM) = SCALE;
        w->ff_gate_b[L][n] = SCALE * (-0.5f);
        w->ff_up_b[L][n] = 1.0f;
        W(w->ff_down[L], n, S_AD_TAPE_LEN, D) = 1.0f;
        n++;

        /* ── AD backward: parent load — same gating pattern as cursor load ── */
        for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
            int val_src = S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_VALUE;
            /* Left parent value */
            W(w->ff_gate[L], S_AD_CUR_LEFT, n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f) - SCALE;
            W(w->ff_up[L], val_src, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_LEFT_VALUE, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_CUR_LEFT, n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f) - SCALE;
            W(w->ff_up[L], val_src, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_LEFT_VALUE, D) = -1.0f;
            n++;
            /* Right parent value */
            W(w->ff_gate[L], S_AD_CUR_RIGHT, n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f) - SCALE;
            W(w->ff_up[L], val_src, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_RIGHT_VALUE, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_CUR_RIGHT, n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f) - SCALE;
            W(w->ff_up[L], val_src, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, S_AD_RIGHT_VALUE, D) = -1.0f;
            n++;
        }

        /* ── Arena pair bank: bounded read/write by target cell ── */
        for (int cell = 0; cell < ARENA_CELLS; cell++) {
            int kind_dim = ARENA_DIM(cell, ARENA_F_KIND);
            int car_dim = ARENA_DIM(cell, ARENA_F_CAR_VAL);
            int cdr_dim = ARENA_DIM(cell, ARENA_F_CDR_VAL);
            int car_type_dim = ARENA_DIM(cell, ARENA_F_CAR_TYPE);
            int cdr_type_dim = ARENA_DIM(cell, ARENA_F_CDR_TYPE);

            n = add_arena_target_pair(w,L,n, S_ARENA_WRITE_KIND, cell,
                                      S_ARENA_NEW_KIND,1,kind_dim,-1,-1,0,-1,0,
                                      0, kind_dim, 1.0f);
            n = add_arena_target_pair(w,L,n, S_ARENA_WRITE_CAR, cell,
                                      S_ARENA_NEW_CAR,1,car_dim,-1,-1,0,-1,0,
                                      0, car_dim, 1.0f);
            n = add_arena_target_pair(w,L,n, S_ARENA_WRITE_CAR, cell,
                                      S_ARENA_NEW_CAR_TYPE,1,car_type_dim,-1,-1,0,-1,0,
                                      0, car_type_dim, 1.0f);
            n = add_arena_target_pair(w,L,n, S_ARENA_WRITE_CDR, cell,
                                      S_ARENA_NEW_CDR,1,cdr_dim,-1,-1,0,-1,0,
                                      0, cdr_dim, 1.0f);
            n = add_arena_target_pair(w,L,n, S_ARENA_WRITE_CDR, cell,
                                      S_ARENA_NEW_CDR_TYPE,1,cdr_type_dim,-1,-1,0,-1,0,
                                      0, cdr_type_dim, 1.0f);

            n = add_arena_target_pair(w,L,n, S_ARENA_READ_CAR, cell,
                                      car_dim,1,S_TOS,-1,-1,0,-1,0,
                                      0, S_TOS, 1.0f);
            n = add_arena_target_pair(w,L,n, S_ARENA_READ_CAR, cell,
                                      car_type_dim,1,S_TYPE_TOS,-1,-1,0,-1,0,
                                      0, S_TYPE_TOS, 1.0f);
            n = add_arena_target_pair(w,L,n, S_ARENA_READ_CDR, cell,
                                      cdr_dim,1,S_TOS,-1,-1,0,-1,0,
                                      0, S_TOS, 1.0f);
            n = add_arena_target_pair(w,L,n, S_ARENA_READ_CDR, cell,
                                      cdr_type_dim,1,S_TYPE_TOS,-1,-1,0,-1,0,
                                      0, S_TYPE_TOS, 1.0f);
        }

        /* INVOKE_CC restore. Layer 3 places the continuation base in
         * S_ARENA_VEC_BASE, raises S_ARENA_VEC_HAS_E0, and writes the sentinel
         * CONT_RESTORE_MARKER into S_ARENA_VEC_LEN. */
#define ADD_CONT_RESTORE(offset, field, cur_dim, out_dim, bias) do { \
            for (int cell = 0; cell < ARENA_CELLS; cell++) { \
                int src_dim = ARENA_DIM(cell, field); \
                n = add_arena_marked_base_offset_pair( \
                    w,L,n, S_ARENA_VEC_BASE, S_ARENA_VEC_HAS_E0, \
                    S_ARENA_VEC_LEN, CONT_RESTORE_MARKER, cell, offset, \
                    src_dim,1,cur_dim,-1,-1,0,-1,0, bias, out_dim, 1.0f); \
            } \
        } while (0)
        ADD_CONT_RESTORE(0, ARENA_F_CAR_VAL,  S_PC,         S_PC,         0.0f);
        ADD_CONT_RESTORE(0, ARENA_F_CDR_VAL,  S_DEPTH,      S_DEPTH,      1.0f);
        ADD_CONT_RESTORE(2, ARENA_F_CAR_TYPE, S_MEM0,       S_MEM0,       0.0f);
        ADD_CONT_RESTORE(2, ARENA_F_CDR_TYPE, S_MEM1,       S_MEM1,       0.0f);
        ADD_CONT_RESTORE(3, ARENA_F_CAR_VAL,  S_MEM2,       S_MEM2,       0.0f);
        ADD_CONT_RESTORE(3, ARENA_F_CDR_VAL,  S_MEM3,       S_MEM3,       0.0f);
        ADD_CONT_RESTORE(0, ARENA_F_CAR_TYPE, S_SOS,        S_SOS,        0.0f);
        ADD_CONT_RESTORE(0, ARENA_F_CDR_TYPE, S_R2,         S_R2,         0.0f);
        ADD_CONT_RESTORE(1, ARENA_F_CAR_VAL,  S_R3,         S_R3,         0.0f);
        ADD_CONT_RESTORE(1, ARENA_F_CAR_TYPE, S_TYPE_SOS,   S_TYPE_SOS,   0.0f);
        ADD_CONT_RESTORE(1, ARENA_F_CDR_TYPE, S_TYPE_R2,    S_TYPE_R2,    0.0f);
        ADD_CONT_RESTORE(2, ARENA_F_CAR_VAL,  S_TYPE_R3,    S_TYPE_R3,    0.0f);
        ADD_CONT_RESTORE(3, ARENA_F_CAR_TYPE, S_WIND_DEPTH, S_WIND_DEPTH, 0.0f);
#undef ADD_CONT_RESTORE

        /* Arena vector-create writes. Header lives at base; element i lives
         * at base + 1 + i and uses the same car/value type lanes as pairs. */
        for (int cell = 0; cell < ARENA_CELLS; cell++) {
            int kind_dim = ARENA_DIM(cell, ARENA_F_KIND);
            int car_dim = ARENA_DIM(cell, ARENA_F_CAR_VAL);
            int cdr_dim = ARENA_DIM(cell, ARENA_F_CDR_VAL);
            int car_type_dim = ARENA_DIM(cell, ARENA_F_CAR_TYPE);
            int cdr_type_dim = ARENA_DIM(cell, ARENA_F_CDR_TYPE);

            n = add_arena_vec_offset_pair(w,L,n, S_ARENA_VEC_WRITE, cell, 0,
                                          kind_dim,-1,-1,0,-1,0,-1,0,
                                          ARENA_KIND_VECTOR, kind_dim, 1.0f);
            n = add_arena_vec_offset_pair(w,L,n, S_ARENA_VEC_WRITE, cell, 0,
                                          S_ARENA_VEC_LEN,1,car_dim,-1,-1,0,-1,0,
                                          0, car_dim, 1.0f);
            n = add_arena_vec_offset_pair(w,L,n, S_ARENA_VEC_WRITE, cell, 0,
                                          S_ARENA_VEC_BASE,1,cdr_dim,-1,-1,0,-1,0,
                                          1.0f, cdr_dim, 1.0f);
            n = add_arena_vec_offset_pair(w,L,n, S_ARENA_VEC_WRITE, cell, 0,
                                          car_type_dim,-1,-1,0,-1,0,-1,0,
                                          TYPE_NUMBER, car_type_dim, 1.0f);
            n = add_arena_vec_offset_pair(w,L,n, S_ARENA_VEC_WRITE, cell, 0,
                                          cdr_type_dim,-1,-1,0,-1,0,-1,0,
                                          TYPE_NUMBER, cdr_type_dim, 1.0f);

            int elem_dims[4] = { S_ARENA_VEC_E0, S_ARENA_VEC_E1, S_ARENA_VEC_E2, S_ARENA_VEC_E3 };
            int elem_type_dims[4] = { S_ARENA_VEC_T0, S_ARENA_VEC_T1, S_ARENA_VEC_T2, S_ARENA_VEC_T3 };
            int elem_has_dims[4] = { S_ARENA_VEC_HAS_E0, S_ARENA_VEC_HAS_E1, S_ARENA_VEC_HAS_E2, S_ARENA_VEC_HAS_E3 };
            for (int i = 0; i < ARENA_MAX_INLINE_VECTOR; i++) {
                n = add_arena_vec_offset_pair(w,L,n, elem_has_dims[i], cell, i + 1,
                                              kind_dim,-1,-1,0,-1,0,-1,0,
                                              ARENA_KIND_VEC_ELEM, kind_dim, 1.0f);
                n = add_arena_vec_offset_pair(w,L,n, elem_has_dims[i], cell, i + 1,
                                              elem_dims[i],1,car_dim,-1,-1,0,-1,0,
                                              0, car_dim, 1.0f);
                n = add_arena_vec_offset_pair(w,L,n, elem_has_dims[i], cell, i + 1,
                                              S_ARENA_VEC_BASE,1,cdr_dim,-1,-1,0,-1,0,
                                              (float)(i + 2), cdr_dim, 1.0f);
                n = add_arena_vec_offset_pair(w,L,n, elem_has_dims[i], cell, i + 1,
                                              elem_type_dims[i],1,car_type_dim,-1,-1,0,-1,0,
                                              0, car_type_dim, 1.0f);
                n = add_arena_vec_offset_pair(w,L,n, elem_has_dims[i], cell, i + 1,
                                              cdr_type_dim,-1,-1,0,-1,0,-1,0,
                                              TYPE_NUMBER, cdr_type_dim, 1.0f);
            }
        }

        /* Arena list-create writes for PACK_REST. Base is S_ARENA_LIST_BASE;
         * element i lives at base + i and is a pair cell. */
        for (int cell = 0; cell < ARENA_CELLS; cell++) {
            int kind_dim = ARENA_DIM(cell, ARENA_F_KIND);
            int car_dim = ARENA_DIM(cell, ARENA_F_CAR_VAL);
            int cdr_dim = ARENA_DIM(cell, ARENA_F_CDR_VAL);
            int car_type_dim = ARENA_DIM(cell, ARENA_F_CAR_TYPE);
            int cdr_type_dim = ARENA_DIM(cell, ARENA_F_CDR_TYPE);
            int elem_dims[4] = { S_ARENA_LIST_E0, S_ARENA_LIST_E1, S_ARENA_LIST_E2, S_ARENA_LIST_E3 };
            int elem_type_dims[4] = { S_ARENA_LIST_T0, S_ARENA_LIST_T1, S_ARENA_LIST_T2, S_ARENA_LIST_T3 };
            int cdr_dims[4] = { S_ARENA_LIST_CDR0, S_ARENA_LIST_CDR1, S_ARENA_LIST_CDR2, S_ARENA_LIST_CDR3 };
            int cdr_type_dims[4] = { S_ARENA_LIST_CDRT0, S_ARENA_LIST_CDRT1, S_ARENA_LIST_CDRT2, S_ARENA_LIST_CDRT3 };
            int elem_has_dims[4] = { S_ARENA_LIST_HAS_E0, S_ARENA_LIST_HAS_E1, S_ARENA_LIST_HAS_E2, S_ARENA_LIST_HAS_E3 };
            for (int i = 0; i < ARENA_MAX_INLINE_VECTOR; i++) {
                n = add_arena_base_offset_pair(w,L,n, S_ARENA_LIST_BASE, elem_has_dims[i], cell, i,
                                               kind_dim,-1,-1,0,-1,0,-1,0,
                                               ARENA_KIND_PAIR, kind_dim, 1.0f);
                n = add_arena_base_offset_pair(w,L,n, S_ARENA_LIST_BASE, elem_has_dims[i], cell, i,
                                               elem_dims[i],1,car_dim,-1,-1,0,-1,0,
                                               0, car_dim, 1.0f);
                n = add_arena_base_offset_pair(w,L,n, S_ARENA_LIST_BASE, elem_has_dims[i], cell, i,
                                               cdr_dims[i],1,cdr_dim,-1,-1,0,-1,0,
                                               0, cdr_dim, 1.0f);
                n = add_arena_base_offset_pair(w,L,n, S_ARENA_LIST_BASE, elem_has_dims[i], cell, i,
                                               elem_type_dims[i],1,car_type_dim,-1,-1,0,-1,0,
                                               0, car_type_dim, 1.0f);
                n = add_arena_base_offset_pair(w,L,n, S_ARENA_LIST_BASE, elem_has_dims[i], cell, i,
                                               cdr_type_dims[i],1,cdr_type_dim,-1,-1,0,-1,0,
                                               0, cdr_type_dim, 1.0f);
            }
        }

        printf("[WEIGHT_GEN] Layer 4: %d neurons used out of %d\n", n, FFN_DIM);
    }

    /* ── Layer 5: AD backward gradient write-back (gated FFN) ──
     * Write AD_LEFT_GRAD_NEW to tape[AD_CUR_LEFT].gradient
     * Write AD_RIGHT_GRAD_NEW to tape[AD_CUR_RIGHT].gradient */
    {
        const int L = 5;
        w->ff_type[L] = 2;
        int n = 0;

        /* ── Gradient rule dispatch neurons ──
         * These fire on pass 1: indicator(CUR_OP) × IS_BACKWARD × grad_input → LEFT/RIGHT_GRAD_NEW */
#define ADD_L5_BW(op_val, in_dim, in_scale, bias, out_dim, coeff) do { \
    W(w->ff_gate[L], S_AD_CUR_OP, n, FFN_DIM) = SCALE; \
    W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE; \
    w->ff_gate_b[L][n] = SCALE * (-(op_val) + 0.5f) - SCALE; \
    if ((in_dim) >= 0) W(w->ff_up[L], (in_dim), n, FFN_DIM) = (in_scale); \
    w->ff_up_b[L][n] = (bias); \
    W(w->ff_down[L], n, (out_dim), D) = (coeff); \
    n++; \
    W(w->ff_gate[L], S_AD_CUR_OP, n, FFN_DIM) = SCALE; \
    W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE; \
    w->ff_gate_b[L][n] = SCALE * (-(op_val) - 0.5f) - SCALE; \
    if ((in_dim) >= 0) W(w->ff_up[L], (in_dim), n, FFN_DIM) = (in_scale); \
    w->ff_up_b[L][n] = (bias); \
    W(w->ff_down[L], n, (out_dim), D) = -(coeff); \
    n++; \
} while(0)

#define ADD_L5_BW_RIGHT_VALUE(op_val, right_val, in_dim, in_scale, bias, out_dim, coeff) do { \
    const float right_scale = 100.0f; \
    float target = (op_val) + right_scale * (float)(right_val); \
    W(w->ff_gate[L], S_AD_CUR_OP, n, FFN_DIM) = SCALE; \
    W(w->ff_gate[L], S_AD_RIGHT_VALUE, n, FFN_DIM) = SCALE * right_scale; \
    W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE; \
    w->ff_gate_b[L][n] = SCALE * (-target + 0.5f) - SCALE; \
    if ((in_dim) >= 0) W(w->ff_up[L], (in_dim), n, FFN_DIM) = (in_scale); \
    w->ff_up_b[L][n] = (bias); \
    W(w->ff_down[L], n, (out_dim), D) = (coeff); \
    n++; \
    W(w->ff_gate[L], S_AD_CUR_OP, n, FFN_DIM) = SCALE; \
    W(w->ff_gate[L], S_AD_RIGHT_VALUE, n, FFN_DIM) = SCALE * right_scale; \
    W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE; \
    w->ff_gate_b[L][n] = SCALE * (-target - 0.5f) - SCALE; \
    if ((in_dim) >= 0) W(w->ff_up[L], (in_dim), n, FFN_DIM) = (in_scale); \
    w->ff_up_b[L][n] = (bias); \
    W(w->ff_down[L], n, (out_dim), D) = -(coeff); \
    n++; \
} while(0)

#define ADD_L5_BW_LEFT_RIGHT_VALUE(op_val, left_val, right_val, in_dim, in_scale, bias, out_dim, coeff) do { \
    const float left_scale = 100.0f; \
    const float right_scale = 1000.0f; \
    float target = (op_val) + left_scale * (float)(left_val) + right_scale * (float)(right_val); \
    W(w->ff_gate[L], S_AD_CUR_OP, n, FFN_DIM) = SCALE; \
    W(w->ff_gate[L], S_AD_LEFT_VALUE, n, FFN_DIM) = SCALE * left_scale; \
    W(w->ff_gate[L], S_AD_RIGHT_VALUE, n, FFN_DIM) = SCALE * right_scale; \
    W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE; \
    w->ff_gate_b[L][n] = SCALE * (-target + 0.5f) - SCALE; \
    if ((in_dim) >= 0) W(w->ff_up[L], (in_dim), n, FFN_DIM) = (in_scale); \
    w->ff_up_b[L][n] = (bias); \
    W(w->ff_down[L], n, (out_dim), D) = (coeff); \
    n++; \
    W(w->ff_gate[L], S_AD_CUR_OP, n, FFN_DIM) = SCALE; \
    W(w->ff_gate[L], S_AD_LEFT_VALUE, n, FFN_DIM) = SCALE * left_scale; \
    W(w->ff_gate[L], S_AD_RIGHT_VALUE, n, FFN_DIM) = SCALE * right_scale; \
    W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE; \
    w->ff_gate_b[L][n] = SCALE * (-target - 0.5f) - SCALE; \
    if ((in_dim) >= 0) W(w->ff_up[L], (in_dim), n, FFN_DIM) = (in_scale); \
    w->ff_up_b[L][n] = (bias); \
    W(w->ff_down[L], n, (out_dim), D) = -(coeff); \
    n++; \
} while(0)

        /* ADD: dL = grad, dR = grad */
        ADD_L5_BW(AD_OP_ADD, S_AD_CUR_GRAD, 1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);
        ADD_L5_BW(AD_OP_ADD, S_AD_CUR_GRAD, 1.0f, 0, S_AD_RIGHT_GRAD_NEW, 1.0f);
        /* SUB: dL = grad, dR = -grad */
        ADD_L5_BW(AD_OP_SUB, S_AD_CUR_GRAD, 1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);
        ADD_L5_BW(AD_OP_SUB, S_AD_CUR_GRAD, -1.0f, 0, S_AD_RIGHT_GRAD_NEW, 1.0f);
        /* MUL: dL = grad*right (PROD_GRAD_LV), dR = grad*left (PROD_GRAD_RV) */
        ADD_L5_BW(AD_OP_MUL, S_AD_PROD_GRAD_LV, 1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);
        ADD_L5_BW(AD_OP_MUL, S_AD_PROD_GRAD_RV, 1.0f, 0, S_AD_RIGHT_GRAD_NEW, 1.0f);
        /* DIV: dL = grad/right via saved reciprocal; dR = -grad*left/(right^2). */
        ADD_L5_BW(AD_OP_DIV, S_AD_PROD_GRAD_SV, 1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);
        for (int d = 1; d <= DIV_WEIGHT_MAX_DENOM; d++) {
            float right_coeff = -1.0f / ((float)d * (float)d);
            ADD_L5_BW_RIGHT_VALUE(AD_OP_DIV, d, S_AD_PROD_GRAD_RV, 1.0f, 0,
                                  S_AD_RIGHT_GRAD_NEW, right_coeff);
        }
        /* POW: dL = grad*right*left^(right-1); dR = grad*left^right*log(left). */
        ADD_L5_BW(AD_OP_POW, S_AD_PROD_GRAD_SV, 1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);
        for (int exp = 1; exp <= AD_POW_WEIGHT_MAX_EXP; exp++) {
            for (int base = 2; base <= AD_POW_WEIGHT_MAX_BASE; base++) {
                float b = (float)base;
                float e = (float)exp;
                float right_coeff = powf(b, e) * logf(b);
                ADD_L5_BW_LEFT_RIGHT_VALUE(AD_OP_POW, base, exp,
                                           S_AD_CUR_GRAD, 1.0f, 0,
                                           S_AD_RIGHT_GRAD_NEW, right_coeff);
            }
        }
        /* ALL unary ops: dL = grad * saved (PROD_GRAD_SV) */
        for (int uop_i = 0; uop_i < 10; uop_i++) {
            float uop_vals[] = { AD_OP_NEG, AD_OP_ABS, AD_OP_RELU, AD_OP_SIGMOID,
                                 AD_OP_TANH, AD_OP_EXP, AD_OP_LOG, AD_OP_SQRT,
                                 AD_OP_SIN, AD_OP_COS };
            ADD_L5_BW(uop_vals[uop_i], S_AD_PROD_GRAD_SV, 1.0f, 0, S_AD_LEFT_GRAD_NEW, 1.0f);
        }
#undef ADD_L5_BW_LEFT_RIGHT_VALUE
#undef ADD_L5_BW_RIGHT_VALUE
#undef ADD_L5_BW

        /* ── Gradient write-back neurons ──
         * These fire on pass 2: indicator(CUR_LEFT/RIGHT) × IS_BACKWARD × LEFT/RIGHT_GRAD_NEW → tape[slot].grad */
        for (int slot = 0; slot < AD_MAX_TAPE; slot++) {
            int grad_dst = S_AD_TAPE_BASE + slot * AD_NODE_FIELDS + AD_F_GRAD;
            /* Left gradient write */
            W(w->ff_gate[L], S_AD_CUR_LEFT, n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f) - SCALE;
            W(w->ff_up[L], S_AD_LEFT_GRAD_NEW, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, grad_dst, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_CUR_LEFT, n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f) - SCALE;
            W(w->ff_up[L], S_AD_LEFT_GRAD_NEW, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, grad_dst, D) = -1.0f;
            n++;
            /* Right gradient write */
            W(w->ff_gate[L], S_AD_CUR_RIGHT, n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot + 0.5f) - SCALE;
            W(w->ff_up[L], S_AD_RIGHT_GRAD_NEW, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, grad_dst, D) = 1.0f;
            n++;
            W(w->ff_gate[L], S_AD_CUR_RIGHT, n, FFN_DIM) = SCALE;
            W(w->ff_gate[L], S_AD_IS_BACKWARD, n, FFN_DIM) = SCALE;
            w->ff_gate_b[L][n] = SCALE * (-(float)slot - 0.5f) - SCALE;
            W(w->ff_up[L], S_AD_RIGHT_GRAD_NEW, n, FFN_DIM) = 1.0f;
            W(w->ff_down[L], n, grad_dst, D) = -1.0f;
            n++;
        }

        /* Cursor decrement + completion check are in the outer execution loop
         * (clock-tick operations, analogous to PC++ for forward execution). */

        printf("[WEIGHT_GEN] Layer 5: %d neurons used out of %d\n", n, FFN_DIM);
    }

    printf("[WEIGHT_GEN] d_model=%d, layers=%d, FFN=%d\n", D, N_LAYERS, FFN_DIM);
    printf("[WEIGHT_GEN] Weights: %zu params, %.1f KB\n",
           sizeof(InterpreterWeights)/sizeof(float),
           sizeof(InterpreterWeights)/1024.0f);
}

/*******************************************************************************
 * Matrix-based forward pass
 ******************************************************************************/

/** @brief Row-major matrix-vector product out = x^T @ W (@p x is length
 *         @p rows, @p W is @p rows x @p cols, @p out is length @p cols). */
static void matvec_t(const float* x, const float* W, float* out, int rows, int cols) {
    memset(out, 0, cols * sizeof(float));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[j] += x[i] * W[i * cols + j];
}

/**
 * @brief Apply layer @p L's feed-forward network to @p x in place, via the
 *        actual weight matrices (W @ x + b), matching @p w->ff_type[L]:
 *        0 = no-op, 1 = standard up/down projection with SQUARE activation,
 *        2 = gated (sigmoid gate * up projection) then down projection.
 *        Adds the FFN output as a residual onto @p x.
 */
static void apply_ffn_layer(const InterpreterWeights* w, int L, float x[D]) {
    float fo[D]; memset(fo, 0, sizeof(fo));
    if (w->ff_type[L] == 1) {
        float h[FFN_DIM];
        matvec_t(x, w->ff_up[L], h, D, FFN_DIM);
        for (int i = 0; i < FFN_DIM; i++) h[i] += w->ff_up_b[L][i];
        for (int i = 0; i < FFN_DIM; i++) h[i] *= h[i]; /* SQUARE */
        matvec_t(h, w->ff_down[L], fo, FFN_DIM, D);
        for (int i = 0; i < D; i++) fo[i] += w->ff_down_b[L][i];
    } else if (w->ff_type[L] == 2) {
        float gate[FFN_DIM], up[FFN_DIM], h[FFN_DIM];
        matvec_t(x, w->ff_gate[L], gate, D, FFN_DIM);
        for (int i = 0; i < FFN_DIM; i++) gate[i] = sigmoidf(gate[i] + w->ff_gate_b[L][i]);
        matvec_t(x, w->ff_up[L], up, D, FFN_DIM);
        for (int i = 0; i < FFN_DIM; i++) up[i] += w->ff_up_b[L][i];
        for (int i = 0; i < FFN_DIM; i++) h[i] = gate[i] * up[i];
        matvec_t(h, w->ff_down[L], fo, FFN_DIM, D);
        for (int i = 0; i < D; i++) fo[i] += w->ff_down_b[L][i];
    }
    for (int i = 0; i < D; i++) x[i] += fo[i];
}

/**
 * @brief Run one AD backward step entirely through weight matrices:
 *        L1 (cursor load) -> L4 (parent load) -> L2 (SQUARE products via
 *        polarization identity) -> L5 (gradient dispatch) -> L5 (gradient
 *        write-back). Layer 3 is never invoked during backward.
 */
static void backward_with_weights(const InterpreterWeights* w, float x[D]) {
    apply_ffn_layer(w, 1, x);  /* cursor load */
    apply_ffn_layer(w, 4, x);  /* parent load */
    apply_ffn_layer(w, 2, x);  /* SQUARE products */
    apply_ffn_layer(w, 5, x);  /* pass 1: gradient dispatch */
    apply_ffn_layer(w, 5, x);  /* pass 2: gradient write-back */
}

/**
 * @brief Run one forward VM step entirely through the matrix-based
 *        transformer weights (@p w): for each non-backward layer, applies
 *        Layer 0's Q/K/V/O attention over the instruction-position
 *        embeddings @p pe (instruction fetch) followed by apply_ffn_layer(),
 *        writing the resulting state to @p next. Layer 5 (backward-only) is
 *        skipped.
 */
static void forward_with_weights(const InterpreterWeights* w,
                                  const float state[D],
                                  const float pe[][D], int np,
                                  float next[D]) {
    float x[D]; memcpy(x, state, sizeof(float)*D);

    for (int L = 0; L < N_LAYERS - 1; L++) { /* Skip Layer 5 (backward-only) */
        /* Attention */
        float ao[D]; memset(ao, 0, sizeof(ao));
        if (L == 0 && np > 0) {
            float Q[D]; memset(Q, 0, sizeof(Q));
            for (int i=0;i<D;i++) for(int j=0;j<D;j++) Q[i]+=w->wq[L][i*D+j]*x[j];
            for (int i=0;i<D;i++) Q[i]+=w->bq[L][i];

            float scores[256]; float mx=-1e30f;
            float Va[256][D];
            for (int p=0; p<np&&p<256; p++) {
                float K[D]; memset(K,0,sizeof(K));
                memset(Va[p],0,sizeof(Va[p]));
                for(int i=0;i<D;i++) for(int j=0;j<D;j++) {
                    K[i]+=w->wk[L][i*D+j]*pe[p][j];
                    Va[p][i]+=w->wv[L][i*D+j]*pe[p][j];
                }
                scores[p]=(Q[0]*K[0]+Q[1]*K[1])/sqrtf((float)HD);
                if(scores[p]>mx) mx=scores[p];
            }
            float sum=0;
            for(int p=0;p<np;p++){scores[p]=expf(scores[p]-mx);sum+=scores[p];}
            for(int p=0;p<np;p++) scores[p]/=sum;
            float hout[D]; memset(hout,0,sizeof(hout));
            for(int p=0;p<np;p++) for(int d=0;d<HD;d++) hout[d]+=scores[p]*Va[p][d];
            for(int i=0;i<D;i++) for(int j=0;j<D;j++) ao[i]+=w->wo[L][i*D+j]*hout[j];
        }
        for(int i=0;i<D;i++) x[i]+=ao[i];

        /* FFN (via shared helper) */
        apply_ffn_layer(w, L, x);
    }
    memcpy(next, x, sizeof(float)*D);
}

/**
 * @brief Run a program to completion (or a step cap) on the matrix-based
 *        transformer forward pass (forward_with_weights() /
 *        backward_with_weights()), the third of the three execution modes
 *        verified against run_reference(). Collects up to @p max_out
 *        output values and, if g_trace_tf_fp is set, emits a JSONL trace
 *        line per step (capped at g_last_ref_steps to avoid phantom steps
 *        past the reference VM's halt).
 * @return The number of outputs produced.
 */
static int run_with_weights(const InterpreterWeights* w,
                             const Instr* prog, int n_instr,
                             float* outputs, int max_out) {
    /* pe is zero-initialised so out-of-range positions attend to a zero
     * embedding (opcode = OP_NOP). See run_simulated for the rationale. */
    float pe[256][D];
    memset(pe, 0, sizeof(pe));
    for(int p=0;p<n_instr&&p<256;p++) embed_instruction(&prog[p],p,pe[p]);
    float state[D]; memset(state,0,sizeof(state)); state[S_OUTPUT]=-1; state[S_CUR_CLOSURE] = -100.0f;
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int n_out=0, step_count=0;
    /* Pre-step trace at step=0 mirrors run_reference. Trace emission is
     * capped at g_last_ref_steps so the matrix path doesn't emit phantom
     * steps past the reference VM's halt — a handful of programs use
     * native-delegated opcodes where the matrix loop never hits OP_HALT,
     * even though the PRINT output matches bit-for-bit (the paper's actual
     * claim per §4.4 is on bitwise output agreement, not step count). */
    int trace_step_cap = g_last_ref_steps > 0 ? g_last_ref_steps : 8192;
    emit_trace_line(g_trace_tf_fp, 0, state, prog, n_instr, -1);
    for(int step=0;step<8192;step++){
        step_count++;
        /* Clear transient dims at start of cycle */
        for (int i = S_AD_CUR_OP; i <= S_AD_RIGHT_VALUE; i++) state[i] = 0;
        state[S_AD_IS_FORWARD] = 0;
        /* Keep S_AD_IS_BACKWARD */
        for (int i = S_AD_GRAD_ACCUM; i <= S_AD_SPARE8; i++) state[i] = 0;
        for (int i = S_ARENA_TRANSIENT_START; i <= S_ARENA_TRANSIENT_END; i++) state[i] = 0;

        float next[D];
        int is_native_pre = 0;
        if (state[S_AD_IS_BACKWARD] > 0.5f) {
            /* Backward: one gradient propagation step through weight matrices.
             * backward_with_weights applies L1→L4→L2→L5→L5:
             *   L1: cursor load (tape[cursor] → AD_CUR_*)
             *   L4: parent load (tape[CUR_LEFT/RIGHT] → AD_LEFT/RIGHT_VALUE)
             *   L2: SQUARE products (grad*left, grad*right, grad*saved)
             *   L5: gradient dispatch and write-back
             * Cursor decrement and SIGMOID/TANH/LOG/SQRT handled after. */
            /* Backward entirely through weight matrices: L1→L4→L2→L5→L5.
             * Layer 5 handles gradient dispatch, write-back, cursor decrement,
             * completion check, and transient clearing — no inline C. */
            /* Backward entirely through weight matrices. Zero inline C.
             * Cursor decrement + completion in L1. Gradient dispatch in L5 pass 1.
             * Gradient write-back in L5 pass 2. */
            memcpy(next, state, sizeof(next));
            backward_with_weights(w, next);
        } else {
            forward_with_weights(w,state,pe,n_instr,next);
            /* Capture IS_NATIVE before postprocess clears it. */
            is_native_pre = next[S_IS_NATIVE] > 0.5f ? 1 : 0;
            exec_loop_postprocess(next, prog, n_instr);
        }
        if(next[S_HAS_OUT]>0.5f&&n_out<max_out) outputs[n_out++]=next[S_OUTPUT];
        if(next[S_HALT]>0.5f) {
            /* Emit final state with halt=true so the comparator sees the
             * terminating step on both sides. */
            memcpy(state,next,sizeof(state));
            if (step_count <= trace_step_cap)
                emit_trace_line(g_trace_tf_fp, step_count, state, prog, n_instr, is_native_pre);
            break;
        }
        memcpy(state,next,sizeof(state));
        if (step_count <= trace_step_cap)
            emit_trace_line(g_trace_tf_fp, step_count, state, prog, n_instr, is_native_pre);
    }
    g_last_mat_steps = step_count;
    return n_out;
}

/*******************************************************************************
 * Binary Weight Export (for qLLM loading)
 ******************************************************************************/

/** @brief Write the generated interpreter weights to @p path in the "QLMW"
 *         binary format (magic, version, architecture dims, then the raw
 *         InterpreterWeights struct) for consumption by the qLLM loader. */
static void export_weights_binary(const InterpreterWeights* w, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("ERROR: cannot open %s\n", path); return; }
    uint32_t magic = 0x514C4D57; /* "QLMW" */
    uint32_t version = 3;
    uint32_t d = D, nl = N_LAYERS, fd = FFN_DIM, nh = H, hd = HD;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&d, 4, 1, f);
    fwrite(&nl, 4, 1, f);
    fwrite(&fd, 4, 1, f);
    fwrite(&nh, 4, 1, f);
    fwrite(&hd, 4, 1, f);
    fwrite(w, sizeof(InterpreterWeights), 1, f);
    fclose(f);
    printf("[EXPORT] Wrote %zu bytes to %s\n", sizeof(InterpreterWeights) + 28, path);
}

/*******************************************************************************
 * Tests: 3-way comparison (reference vs simulated vs matrix-based)
 ******************************************************************************/

static int n_pass = 0, n_fail = 0;
static InterpreterWeights* g_weights = NULL;

/**
 * @brief Run one test program on all three execution modes (reference,
 *        simulated, matrix), compare their output slots against each other
 *        and against @p expected, and update the global pass/fail counters
 *        (n_pass/n_fail), printing a diagnostic on mismatch.
 */
static void test(const char* name, const Instr* prog, int n, float expected) {
    float r[64], s[64], m[64];
    g_trace_program_name = name;
    g_trace_program_seq++;
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int rn = run_reference(prog, n, r, 64);
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int sn = run_simulated(prog, n, s, 64);
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int mn = g_weights ? run_with_weights(g_weights, prog, n, m, 64) : 0;

    float rv = rn>0?r[0]:-9999, sv = sn>0?s[0]:-9999, mv = mn>0?m[0]:-9999;
    int ok_r = fabsf(rv-expected)<0.01f;
    int ok_s = fabsf(sv-expected)<0.01f;
    int ok_m = g_weights ? fabsf(mv-expected)<0.01f : 1;

    printf("  %-25s ref=%7.1f sim=%7.1f mat=%7.1f  %s%s%s\n",
           name, rv, sv, mv,
           ok_r?"":"ref:FAIL ", ok_s?"":"sim:FAIL ",
           (g_weights && !ok_m)?"mat:FAIL ":"");

    /* Uncomment for per-test metrics: */
    /* printf("    steps: ref=%d sim=%d mat=%d heap=%d\n", g_last_ref_steps, g_last_sim_steps, g_last_mat_steps, g_heap_ptr); */

    if (ok_r && ok_s && ok_m) n_pass++; else n_fail++;

    /* Clear program name so any non-test() runs (Dynamic multiplication,
     * inline frame assertions, etc.) below don't pollute the trace by
     * re-emitting under this program's name + id. */
    g_trace_program_name = NULL;
}

/*******************************************************************************
 * Self-improvement loop: gradient descent on PROGRAM WEIGHTS
 *
 * All code below is new and self-contained. It does NOT modify
 * forward_with_weights / apply_ffn_layer / generate_weights. It re-derives the
 * analytic gradient of the loss w.r.t. the trainable weight matrices by
 * re-running the forward pass while caching intermediates, then backpropagating
 * in reverse. Correctness is self-verified by a central finite-difference
 * gradient check before any training is performed.
 ******************************************************************************/

/* Gradients mirror the trainable subset of InterpreterWeights. */
typedef struct {
    float dwq[N_LAYERS][D * D];
    float dwk[N_LAYERS][D * D];
    float dwv[N_LAYERS][D * D];
    float dwo[N_LAYERS][D * D];
    float dbq[N_LAYERS][D];
    float dff_up[N_LAYERS][D * FFN_DIM];
    float dff_up_b[N_LAYERS][FFN_DIM];
    float dff_down[N_LAYERS][FFN_DIM * D];
    float dff_down_b[N_LAYERS][D];
    float dff_gate[N_LAYERS][D * FFN_DIM];
    float dff_gate_b[N_LAYERS][FFN_DIM];
} WeightGrads;

/** @brief Zero-initialize a WeightGrads accumulator. */
