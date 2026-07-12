static void zero_grads(WeightGrads* g) {
    memset(g, 0, sizeof(WeightGrads));
}

/* Per-layer forward cache so the backward pass can reuse exact intermediates. */
typedef struct {
    float x_in[N_LAYERS][D];      /* x at the *start* of layer L's iteration   */
    float x_post_attn[N_LAYERS][D]; /* x after the attention residual (== x_in for L>0) */
    /* Attention (only meaningful for L==0, np>0) */
    int   attn_active;
    float Q[D];
    float K[256][D];
    float Va[256][D];
    float scores[256];            /* post-softmax weights                       */
    float hout[D];
    int   np;
    /* FFN intermediates for the active type. */
    float ff_h[N_LAYERS][FFN_DIM];     /* type1: pre-square h (post-bias); type2: gate*up */
    float ff_gate[N_LAYERS][FFN_DIM];  /* type2: sigmoid(.) */
    float ff_up[N_LAYERS][FFN_DIM];    /* type2: matvec+bias (pre-product)       */
} FwdCache;

/**
 * @brief Backprop through one matvec_t() call: out[j] = sum_i
 *        x[i]*W[i*cols+j]. Accumulates dW[i*cols+j] += x[i]*dout[j] and (if
 *        @p dx is non-null) dx[i] += sum_j dout[j]*W[i*cols+j]. @p dx may be
 *        NULL when the input is a constant (e.g. the position embeddings).
 */
static void matvec_backward(const float* x, const float* W, const float* dout,
                            float* dW, float* dx, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float dxi = 0.0f;
        const float xi = x[i];
        const float* Wi = W + (size_t)i * cols;
        float* dWi = dW + (size_t)i * cols;
        for (int j = 0; j < cols; j++) {
            float dj = dout[j];
            dWi[j] += xi * dj;
            dxi += dj * Wi[j];
        }
        if (dx) dx[i] += dxi;
    }
}

/**
 * @brief Re-run forward_with_weights() while caching per-layer
 *        intermediates (attention scores/Q/K/V, FFN gate/up/pre-activation
 *        values), then backpropagate the upstream gradient @p dL_dnext
 *        (grad on `next`) through each layer's FFN and, for Layer 0, its
 *        attention block — analytically re-deriving the loss gradient
 *        w.r.t. every trainable weight matrix without touching
 *        forward_with_weights()/apply_ffn_layer()/generate_weights().
 *        Accumulates gradients into @p g and, if @p dL_dstate is non-null,
 *        writes the gradient w.r.t. the input state there.
 */
static void backward_through_weights(const InterpreterWeights* w,
                                     const float state[D],
                                     const float pe[][D], int np,
                                     const float dL_dnext[D],
                                     WeightGrads* g,
                                     float dL_dstate[D]) {
    static FwdCache c;                 /* large; keep off the stack */
    memset(&c, 0, sizeof(c));
    c.np = np;

    /* ---- forward (mirrors forward_with_weights exactly, but caches) ---- */
    float x[D]; memcpy(x, state, sizeof(float) * D);
    for (int L = 0; L < N_LAYERS - 1; L++) {
        memcpy(c.x_in[L], x, sizeof(float) * D);

        float ao[D]; memset(ao, 0, sizeof(ao));
        if (L == 0 && np > 0) {
            c.attn_active = 1;
            float Q[D]; memset(Q, 0, sizeof(Q));
            for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) Q[i] += w->wq[L][i*D+j]*x[j];
            for (int i = 0; i < D; i++) Q[i] += w->bq[L][i];
            memcpy(c.Q, Q, sizeof(Q));

            float scores[256]; float mx = -1e30f;
            for (int p = 0; p < np && p < 256; p++) {
                float K[D]; memset(K, 0, sizeof(K));
                memset(c.Va[p], 0, sizeof(float)*D);
                for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) {
                    K[i] += w->wk[L][i*D+j]*pe[p][j];
                    c.Va[p][i] += w->wv[L][i*D+j]*pe[p][j];
                }
                memcpy(c.K[p], K, sizeof(float)*D);
                scores[p] = (Q[0]*K[0] + Q[1]*K[1]) / sqrtf((float)HD);
                if (scores[p] > mx) mx = scores[p];
            }
            float sum = 0;
            for (int p = 0; p < np; p++) { scores[p] = expf(scores[p]-mx); sum += scores[p]; }
            for (int p = 0; p < np; p++) scores[p] /= sum;
            memcpy(c.scores, scores, sizeof(float)*np);
            float hout[D]; memset(hout, 0, sizeof(hout));
            for (int p = 0; p < np; p++) for (int d = 0; d < HD; d++) hout[d] += scores[p]*c.Va[p][d];
            memcpy(c.hout, hout, sizeof(hout));
            for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) ao[i] += w->wo[L][i*D+j]*hout[j];
        }
        for (int i = 0; i < D; i++) x[i] += ao[i];
        memcpy(c.x_post_attn[L], x, sizeof(float)*D);

        /* FFN (mirror apply_ffn_layer, caching intermediates) */
        if (w->ff_type[L] == 1) {
            float h[FFN_DIM];
            matvec_t(x, w->ff_up[L], h, D, FFN_DIM);
            for (int i = 0; i < FFN_DIM; i++) h[i] += w->ff_up_b[L][i];
            memcpy(c.ff_h[L], h, sizeof(float)*FFN_DIM);   /* pre-square */
            for (int i = 0; i < FFN_DIM; i++) h[i] *= h[i];
            float fo[D]; matvec_t(h, w->ff_down[L], fo, FFN_DIM, D);
            for (int i = 0; i < D; i++) fo[i] += w->ff_down_b[L][i];
            for (int i = 0; i < D; i++) x[i] += fo[i];
        } else if (w->ff_type[L] == 2) {
            float gate[FFN_DIM], up[FFN_DIM], h[FFN_DIM];
            matvec_t(x, w->ff_gate[L], gate, D, FFN_DIM);
            for (int i = 0; i < FFN_DIM; i++) gate[i] = sigmoidf(gate[i] + w->ff_gate_b[L][i]);
            matvec_t(x, w->ff_up[L], up, D, FFN_DIM);
            for (int i = 0; i < FFN_DIM; i++) up[i] += w->ff_up_b[L][i];
            for (int i = 0; i < FFN_DIM; i++) h[i] = gate[i]*up[i];
            memcpy(c.ff_gate[L], gate, sizeof(float)*FFN_DIM);
            memcpy(c.ff_up[L], up, sizeof(float)*FFN_DIM);
            memcpy(c.ff_h[L], h, sizeof(float)*FFN_DIM);
            float fo[D]; matvec_t(h, w->ff_down[L], fo, FFN_DIM, D);
            for (int i = 0; i < D; i++) fo[i] += w->ff_down_b[L][i];
            for (int i = 0; i < D; i++) x[i] += fo[i];
        }
        /* type 0: x unchanged */
    }

    /* ---- backward ---- */
    float dx[D]; memcpy(dx, dL_dnext, sizeof(float)*D);  /* grad on x after last layer */

    for (int L = N_LAYERS - 2; L >= 0; L--) {
        /* FFN backward: x_out = x_post_attn + fo(x_post_attn).
         * Residual: grad flows to the FFN input through both the branch and the
         * identity skip. dx currently holds grad on x_out. */
        if (w->ff_type[L] == 1) {
            /* fo = matvec_t(h2, ff_down) + ff_down_b ; h2 = h*h ; h = matvec_t(x,ff_up)+ff_up_b */
            float* dfo = dx;                       /* grad on fo == grad on x_out (skip handled below) */
            for (int i = 0; i < D; i++) g->dff_down_b[L][i] += dfo[i];
            float dh2[FFN_DIM]; memset(dh2, 0, sizeof(dh2));
            float h[FFN_DIM];                      /* recompute squared activation */
            for (int i = 0; i < FFN_DIM; i++) h[i] = c.ff_h[L][i]*c.ff_h[L][i];
            matvec_backward(h, w->ff_down[L], dfo, g->dff_down[L], dh2, FFN_DIM, D);
            /* square deriv: dh = dh2 * 2*h_pre */
            float dh[FFN_DIM];
            for (int i = 0; i < FFN_DIM; i++) dh[i] = dh2[i] * 2.0f * c.ff_h[L][i];
            for (int i = 0; i < FFN_DIM; i++) g->dff_up_b[L][i] += dh[i];
            float dxin[D]; memset(dxin, 0, sizeof(dxin));
            matvec_backward(c.x_post_attn[L], w->ff_up[L], dh, g->dff_up[L], dxin, D, FFN_DIM);
            for (int i = 0; i < D; i++) dx[i] += dxin[i]; /* branch grad + skip (already in dx) */
        } else if (w->ff_type[L] == 2) {
            float* dfo = dx;
            for (int i = 0; i < D; i++) g->dff_down_b[L][i] += dfo[i];
            float dh[FFN_DIM]; memset(dh, 0, sizeof(dh));
            matvec_backward(c.ff_h[L], w->ff_down[L], dfo, g->dff_down[L], dh, FFN_DIM, D);
            /* h = gate*up -> dgate = dh*up, dup = dh*gate */
            float dgate[FFN_DIM], dup[FFN_DIM];
            for (int i = 0; i < FFN_DIM; i++) { dgate[i] = dh[i]*c.ff_up[L][i]; dup[i] = dh[i]*c.ff_gate[L][i]; }
            /* up = matvec_t(x,ff_up)+ff_up_b */
            for (int i = 0; i < FFN_DIM; i++) g->dff_up_b[L][i] += dup[i];
            float dxin[D]; memset(dxin, 0, sizeof(dxin));
            matvec_backward(c.x_post_attn[L], w->ff_up[L], dup, g->dff_up[L], dxin, D, FFN_DIM);
            /* gate = sigmoid(pre); dpre = dgate * g*(1-g) */
            float dpre[FFN_DIM];
            for (int i = 0; i < FFN_DIM; i++) { float gv = c.ff_gate[L][i]; dpre[i] = dgate[i]*gv*(1.0f-gv); }
            for (int i = 0; i < FFN_DIM; i++) g->dff_gate_b[L][i] += dpre[i];
            matvec_backward(c.x_post_attn[L], w->ff_gate[L], dpre, g->dff_gate[L], dxin, D, FFN_DIM);
            for (int i = 0; i < D; i++) dx[i] += dxin[i];
        }
        /* type 0: x_out == x_post_attn, dx unchanged */

        /* Attention backward (only L==0 with active attention).
         * x_post_attn = x_in + ao ; dx currently holds grad on x_post_attn.
         * The identity skip contributes dx directly to x_in (kept in dx). */
        if (L == 0 && c.attn_active) {
            /* NOTE: attention in forward_with_weights uses OUTPUT-major weight
             * layout (out[i] = sum_j W[i*D+j]*in[j]), which is the TRANSPOSE of
             * matvec_t's input-major layout. So we backprop with explicit loops
             * here (dW[i*D+j] += in[j]*dout[i]; din[j] += W[i*D+j]*dout[i]). */
            float* dxpa = dx;            /* grad on x_post_attn (== grad on ao for the branch) */
            /* ao[i] = sum_j wo[i*D+j]*hout[j] */
            float dhout[D]; memset(dhout, 0, sizeof(dhout));
            for (int i = 0; i < D; i++) {
                float doi = dxpa[i];
                for (int j = 0; j < D; j++) {
                    g->dwo[0][i*D+j] += c.hout[j]*doi;
                    dhout[j]        += w->wo[0][i*D+j]*doi;
                }
            }
            /* hout[d] = sum_p scores[p]*Va[p][d], d in 0..HD-1 (else 0) */
            int np_ = c.np;
            float ds[256]; memset(ds, 0, sizeof(float)*np_);
            for (int p = 0; p < np_; p++) {
                float acc = 0.0f;
                for (int d = 0; d < HD; d++) acc += dhout[d]*c.Va[p][d];
                ds[p] = acc;                 /* dL/dscore_p (post-softmax) */
                /* Va[p][i] = sum_j wv[i*D+j]*pe[p][j]; only d<HD affect hout */
                for (int d = 0; d < HD; d++) {
                    float dVad = c.scores[p]*dhout[d];   /* dL/dVa[p][d] */
                    for (int j = 0; j < D; j++) g->dwv[0][d*D+j] += pe[p][j]*dVad;
                }
            }
            /* softmax jacobian: dscore_p = s_p*(ds_p - sum_q s_q*ds_q) */
            float dot = 0.0f;
            for (int p = 0; p < np_; p++) dot += c.scores[p]*ds[p];
            float dscore[256];
            for (int p = 0; p < np_; p++) dscore[p] = c.scores[p]*(ds[p] - dot);
            /* score_p = (Q0*K_p0 + Q1*K_p1)/sqrt(HD) */
            float inv = 1.0f/sqrtf((float)HD);
            float dQ[D]; memset(dQ, 0, sizeof(dQ));
            for (int p = 0; p < np_; p++) {
                float dsp = dscore[p]*inv;
                /* dQ0 += dsp*K_p0 ; dQ1 += dsp*K_p1 ; dK_p0 += dsp*Q0 ; dK_p1 += dsp*Q1 */
                dQ[0] += dsp*c.K[p][0];
                dQ[1] += dsp*c.K[p][1];
                float dK0 = dsp*c.Q[0];
                float dK1 = dsp*c.Q[1];
                /* K[p][i] = sum_j wk[i*D+j]*pe[p][j]; only i=0,1 used */
                for (int j = 0; j < D; j++) {
                    g->dwk[0][0*D+j] += pe[p][j]*dK0;
                    g->dwk[0][1*D+j] += pe[p][j]*dK1;
                }
            }
            /* Q[i] = sum_j wq[i*D+j]*x_in[j] + bq[i]; only Q[0],Q[1] used */
            for (int i = 0; i < D; i++) g->dbq[0][i] += dQ[i];
            float dxin[D]; memset(dxin, 0, sizeof(dxin));
            for (int i = 0; i < D; i++) {
                float dqi = dQ[i];
                if (dqi == 0.0f) continue;
                for (int j = 0; j < D; j++) {
                    g->dwq[0][i*D+j] += c.x_in[0][j]*dqi;
                    dxin[j]          += w->wq[0][i*D+j]*dqi;
                }
            }
            for (int i = 0; i < D; i++) dx[i] += dxin[i];  /* branch grad + skip */
        }
    }

    if (dL_dstate) memcpy(dL_dstate, dx, sizeof(float)*D);
}

/** @brief Apply one SGD step to every trainable weight matrix/bias in @p w:
 *         w -= lr * g, for the given learning rate @p lr and accumulated
 *         gradients @p g. */
static void apply_weight_gradient_step(InterpreterWeights* w,
                                       const WeightGrads* g, float lr) {
    for (int L = 0; L < N_LAYERS; L++) {
        for (int i = 0; i < D*D; i++) {
            w->wq[L][i] -= lr*g->dwq[L][i];
            w->wk[L][i] -= lr*g->dwk[L][i];
            w->wv[L][i] -= lr*g->dwv[L][i];
            w->wo[L][i] -= lr*g->dwo[L][i];
        }
        for (int i = 0; i < D; i++) w->bq[L][i] -= lr*g->dbq[L][i];
        for (int i = 0; i < D*FFN_DIM; i++) {
            w->ff_up[L][i]   -= lr*g->dff_up[L][i];
            w->ff_gate[L][i] -= lr*g->dff_gate[L][i];
        }
        for (int i = 0; i < FFN_DIM; i++) {
            w->ff_up_b[L][i]   -= lr*g->dff_up_b[L][i];
            w->ff_gate_b[L][i] -= lr*g->dff_gate_b[L][i];
        }
        for (int i = 0; i < FFN_DIM*D; i++) w->ff_down[L][i] -= lr*g->dff_down[L][i];
        for (int i = 0; i < D; i++) w->ff_down_b[L][i] -= lr*g->dff_down_b[L][i];
    }
}

/**
 * @brief Double-precision reference forward pass — a faithful,
 *        higher-precision mirror of forward_with_weights() (same
 *        arithmetic, same order) plus a squared-error loss against
 *        @p target. Used ONLY by the gradient check's finite-difference
 *        loss, since a float32 central difference is unreliable for
 *        small-gradient attention parameters (verified: float32 FD
 *        disagrees with both the analytic gradient and a double FD by
 *        ~13%, while the double FD matches the analytic gradient to 6
 *        digits). Does not modify the artifact's float32 forward pass.
 * @return The squared-error loss 0.5*sum((x_i - target_i)^2).
 */
static double forward_loss_double(const InterpreterWeights* w,
                                  const float state[D], const float pe[][D],
                                  int np, const double target[D]) {
    double x[D]; for (int i = 0; i < D; i++) x[i] = state[i];
    for (int L = 0; L < N_LAYERS - 1; L++) {
        double ao[D]; for (int i = 0; i < D; i++) ao[i] = 0.0;
        if (L == 0 && np > 0) {
            double Q[D];
            for (int i = 0; i < D; i++) { Q[i] = 0.0; for (int j = 0; j < D; j++) Q[i] += (double)w->wq[L][i*D+j]*x[j]; Q[i] += w->bq[L][i]; }
            double sc[256], mx = -1e300, Va[256][2];
            for (int p = 0; p < np && p < 256; p++) {
                double K0 = 0.0, K1 = 0.0;
                Va[p][0] = 0.0; Va[p][1] = 0.0;
                for (int j = 0; j < D; j++) {
                    K0 += (double)w->wk[L][0*D+j]*pe[p][j];
                    K1 += (double)w->wk[L][1*D+j]*pe[p][j];
                    Va[p][0] += (double)w->wv[L][0*D+j]*pe[p][j];
                    Va[p][1] += (double)w->wv[L][1*D+j]*pe[p][j];
                }
                sc[p] = (Q[0]*K0 + Q[1]*K1) / sqrt((double)HD);
                if (sc[p] > mx) mx = sc[p];
            }
            double sum = 0.0; for (int p = 0; p < np; p++) { sc[p] = exp(sc[p]-mx); sum += sc[p]; }
            for (int p = 0; p < np; p++) sc[p] /= sum;
            double hout[D]; for (int i = 0; i < D; i++) hout[i] = 0.0;
            for (int p = 0; p < np; p++) for (int d = 0; d < HD; d++) hout[d] += sc[p]*Va[p][d];
            for (int i = 0; i < D; i++) for (int j = 0; j < D; j++) ao[i] += (double)w->wo[L][i*D+j]*hout[j];
        }
        for (int i = 0; i < D; i++) x[i] += ao[i];
        /* FFN, mirroring apply_ffn_layer in double */
        double fo[D]; for (int i = 0; i < D; i++) fo[i] = 0.0;
        if (w->ff_type[L] == 1) {
            double h[FFN_DIM];
            for (int j = 0; j < FFN_DIM; j++) { double s = 0.0; for (int i = 0; i < D; i++) s += x[i]*(double)w->ff_up[L][i*FFN_DIM+j]; h[j] = s + w->ff_up_b[L][j]; h[j] *= h[j]; }
            for (int j = 0; j < D; j++) { double s = 0.0; for (int i = 0; i < FFN_DIM; i++) s += h[i]*(double)w->ff_down[L][i*D+j]; fo[j] = s + w->ff_down_b[L][j]; }
        } else if (w->ff_type[L] == 2) {
            double h[FFN_DIM];
            for (int j = 0; j < FFN_DIM; j++) {
                double sg_ = 0.0, su = 0.0;
                for (int i = 0; i < D; i++) { sg_ += x[i]*(double)w->ff_gate[L][i*FFN_DIM+j]; su += x[i]*(double)w->ff_up[L][i*FFN_DIM+j]; }
                double gate = 1.0/(1.0+exp(-(sg_ + w->ff_gate_b[L][j])));
                h[j] = gate * (su + w->ff_up_b[L][j]);
            }
            for (int j = 0; j < D; j++) { double s = 0.0; for (int i = 0; i < FFN_DIM; i++) s += h[i]*(double)w->ff_down[L][i*D+j]; fo[j] = s + w->ff_down_b[L][j]; }
        }
        for (int i = 0; i < D; i++) x[i] += fo[i];
    }
    double Lo = 0.0; for (int i = 0; i < D; i++) { double d = x[i]-target[i]; Lo += 0.5*d*d; }
    return Lo;
}

/* Tiny deterministic LCG so the demo is reproducible. */
static unsigned long g_si_rng = 0;
static float g_si_scale = 0.1f;
/** @brief Deterministic LCG PRNG (seeded via g_si_rng), returning a value
 *         uniform in [-g_si_scale, g_si_scale] for reproducible weight
