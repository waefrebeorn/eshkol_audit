/**
 * @file weight_matrices.c
 * @brief Universal Eshkol VM interpreter compiled into transformer weights.
 *
 * Full ISA implementation (84 opcodes: 65 base + 19 AD, d_model=256).
 * Opcode numbering matches eshkol_compiler.c canonical enum.
 *
 * Three execution modes verified against each other:
 *   1. Reference interpreter (direct C switch)
 *   2. Simulated transformer (C functions mirroring weight computation)
 *   3. Matrix-based forward pass (actual W @ x + b through generic matmul)
 *
 * Architecture: d_model=256, n_heads=16, head_dim=2, n_layers=6, FFN_DIM=2304
 *   Layer 0: Instruction fetch (Gaussian attention peaked at PC)
 *   Layer 1: Preprocessing (gated FFN for address resolution, comparisons, AD cursor load)
 *   Layer 2: Product precompute (SQUARE activation for TOS*SOS + AD products)
 *   Layer 3: Execution (gated FFN for opcode dispatch + AD forward/backward rules)
 *   Layer 4: Tape write + parent load (gated FFN)
 *   Layer 5: Gradient write-back (gated FFN, backward-only)
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ESKB binary format reader (single-file include pattern) */
#include "eskb_reader.c"

/* Runtime libraries for extended native call dispatch.
 * Note: vm_tensor.c is included transitively via vm_tensor_ops.c,
 *       vm_string.c is included transitively via vm_io.c. */
#include "vm_numeric.h"
#include "vm_complex.c"
#include "vm_rational.c"
#include "vm_bignum.c"
#include "vm_dual.c"
#include "vm_autodiff.c"
#include "vm_tensor_ops.c"
#include "vm_logic.c"
#include "vm_inference.c"
#include "vm_workspace.c"
#include "vm_io.c"
#include "vm_hashtable.c"
#include "vm_bytevector.c"
#include "vm_multivalue.c"
#include "vm_error.c"
#include "vm_parameter.c"

#define D 256
#define H 16
#define HD 2
#define N_LAYERS 6
#define MEM_SIZE 4
#define FFN_DIM 2304
/* Temperature for softmax/sigmoid gates.
 *
 * For bit-identical agreement between the matrix forward pass and the
 * reference C interpreter, every gate must saturate to *exactly* 0 or 1
 * — no sub-ulp leakage that accumulates over thousands of steps.
 *
 * Two saturation conditions:
 *
 *   1. sigmoid(x) gates (used by `indicator()` for opcode/operand dispatch):
 *      sigmoidf() short-circuits to 1.0f for x > 20 and 0.0f for x < -20.
 *      With integer x and k separated by ≥1, the sigmoid argument is
 *      ≥ SCALE/2 in absolute value, so SCALE > 40 already saturates these.
 *
 *   2. Softmax over position embeddings (Layer 0 attention):
 *      Score gap between the peak (p == PC) and adjacent positions is
 *      ½·SCALE/sqrt(HD) = SCALE/(2·sqrt(2)) ≈ 0.3536·SCALE.
 *      For exp(−gap) to underflow to *literal* zero in float32 (denormals
 *      kick in around exp(−87) ≈ 1e-38), we need gap > 87, i.e.
 *      SCALE > 246. Anything below that leaves a residue (with SCALE=100
 *      the peak−adjacent residue is exp(−35.4) ≈ 4.6e-16, which is
 *      perfectly representable in float32 and propagates indefinitely
 *      through accumulation chains — observed as a tos=4.4e-16 in
 *      `tail sum(100)` at step 1206 vs. exactly 0 in the reference).
 *
 * SCALE = 300 satisfies both with margin (gap ≈ 106 > 87, sigmoid
 * argument ≥ 150 ≫ 20). */
#define SCALE 300.0f
#define AD_MAX_TAPE 8    /* max tape nodes in state vector */
#define AD_NODE_FIELDS 8 /* fields per tape node */
#define ARENA_CELLS 16
#define ARENA_CELL_FIELDS 5
#define ARENA_KIND_EMPTY 0.0f
#define ARENA_KIND_PAIR 1.0f
#define ARENA_KIND_VECTOR 2.0f
#define ARENA_KIND_VEC_ELEM 3.0f
#define ARENA_KIND_CLOSURE 4.0f
#define ARENA_MAX_INLINE_VECTOR 4
#define ARENA_CONT_CELLS 4
#define CONT_RESTORE_MARKER 7.0f
#define AD_TRIG_WEIGHT_MIN_INPUT -4
#define AD_TRIG_WEIGHT_MAX_INPUT 4

/* Opcodes — canonical numbering from eshkol_compiler.c */
typedef enum {
    OP_NOP=0, OP_CONST=1, OP_NIL=2, OP_TRUE=3, OP_FALSE=4, OP_POP=5, OP_DUP=6,
    OP_ADD=7, OP_SUB=8, OP_MUL=9, OP_DIV=10, OP_MOD=11, OP_NEG=12, OP_ABS=13,
    OP_EQ=14, OP_LT=15, OP_GT=16, OP_LE=17, OP_GE=18, OP_NOT=19,
    OP_GET_LOCAL=20, OP_SET_LOCAL=21, OP_GET_UPVALUE=22, OP_SET_UPVALUE=23,
    OP_CLOSURE=24, OP_CALL=25, OP_TAIL_CALL=26, OP_RETURN=27,
    OP_JUMP=28, OP_JUMP_IF_FALSE=29, OP_LOOP=30,
    OP_CONS=31, OP_CAR=32, OP_CDR=33, OP_NULL_P=34,
    OP_PRINT=35, OP_HALT=36, OP_NATIVE_CALL=37,
    OP_CLOSE_UPVALUE=38,
    OP_VEC_CREATE=39, OP_VEC_REF=40, OP_VEC_SET=41, OP_VEC_LEN=42,
    OP_STR_REF=43, OP_STR_LEN=44,
    OP_PAIR_P=45, OP_NUM_P=46, OP_STR_P=47, OP_BOOL_P=48, OP_PROC_P=49, OP_VEC_P=50,
    OP_SET_CAR=51, OP_SET_CDR=52, OP_POPN=53,
    OP_OPEN_CLOSURE=54, OP_CALLCC=55, OP_INVOKE_CC=56,
    OP_PUSH_HANDLER=57, OP_POP_HANDLER=58, OP_GET_EXN=59,
    OP_PACK_REST=60, OP_WIND_PUSH=61, OP_WIND_POP=62, OP_VOID=63,

    /* AD opcodes — bounded tape ops are weight-encoded; libm/precision ops delegate. */
    OP_AD_VAR=64, OP_AD_CONST=65,
    OP_AD_ADD=66, OP_AD_SUB=67, OP_AD_MUL=68,
    OP_AD_NEG=69, OP_AD_ABS=70, OP_AD_RELU=71,
    OP_AD_SIGMOID=72, OP_AD_TANH=73,
    OP_AD_EXP=74, OP_AD_LOG=75, OP_AD_SQRT=76,
    OP_AD_BACKWARD=77, OP_AD_GRAD=78,
    /* AD ops delegated to C (transcendentals / division) */
    OP_AD_DIV=79, OP_AD_POW=80, OP_AD_SIN=81, OP_AD_COS=82,

    /* Base stack op appended after the AD band (base band 0-63 is full).
     * SWAP exchanges the top two stack registers (TOS<->SOS). It is a base
     * stack op, NOT an AD op — no AD range check (<= OP_AD_SQRT) sees it. */
    OP_SWAP=83,

    OP_COUNT=84
} OpCode;

typedef struct { OpCode op; int operand; } Instr;

/* State vector layout (d_model=256) */
enum {
    /* Permanent state (0-15) — persist across steps */
    S_PC=0, S_TOS=1, S_SOS=2, S_R2=3, S_R3=4, S_DEPTH=5,
    S_OUTPUT=6, S_HALT=7,
    S_MEM0=8, S_MEM1=9, S_MEM2=10, S_MEM3=11,
    S_SP=12, S_FP=13, S_HAS_OUT=14, S_CUR_CLOSURE=15,
    S_EXC_DEPTH=S_SP, S_WIND_DEPTH=S_FP,

    /* Intermediate / transient (16-31) — cleared every cycle by Layer 3 */
    S_OPCODE=16, S_OPERAND=17,
    S_PRODUCT=18, S_LOADVAL=19,
    S_STORED0=20, S_STORED1=21, S_STORED2=22, S_STORED3=23,
    S_ZOPER=24, S_ZPC1=25,
    S_CMP_EQ=26, S_CMP_LT=27,
    S_IS_CALL=28, S_IS_RET=29, S_IS_NATIVE=30,
    S_ABS_DELTA=31,

    /* Type tags for TOS/SOS/R2/R3 (32-35) — persist across steps.
     * Type encoding: 0=number, 1=boolean, 2=pair, 3=closure,
     *                4=string, 5=vector, 6=nil, 7=continuation */
    S_TYPE_TOS=32, S_TYPE_SOS=33, S_TYPE_R2=34, S_TYPE_R3=35,

    /* ── Zone B: AD control state (36-47) — persist across steps ── */
    S_AD_TAPE_LEN=36,    /* number of nodes on tape (0..AD_MAX_TAPE) */
    S_AD_CURSOR=37,      /* backward pass cursor (current node index, decrements) */
    S_AD_MODE=38,        /* 0=normal, 1=forward recording, 2=backward pass */
    S_AD_CUR_OP=39,      /* operation type of node at cursor */
    S_AD_CUR_VALUE=40,   /* forward value of node at cursor */
    S_AD_CUR_GRAD=41,    /* gradient of node at cursor */
    S_AD_CUR_LEFT=42,    /* left parent index */
    S_AD_CUR_RIGHT=43,   /* right parent index */
    S_AD_CUR_SAVED=44,   /* auxiliary saved value */
    S_AD_LEFT_VALUE=45,  /* value of left parent (loaded for backward) */
    S_AD_LEFT_GRAD=46,   /* gradient of left parent */
    S_AD_UNARY_ABS_ACTIVE=S_AD_LEFT_GRAD, /* forward scratch alias */
    S_AD_RIGHT_VALUE=47, /* value of right parent */

    /* ── Zone C: AD tape storage (48-111) — 8 nodes x 8 fields ──
     * Node i at dims (48 + i*8) through (48 + i*8 + 7)
     * Fields: [op, value, gradient, left, right, saved, spare0, spare1] */
    S_AD_TAPE_BASE=48,
    /* Access macro: S_AD_TAPE_BASE + node_idx * AD_NODE_FIELDS + field_offset */

    /* ── Zone D: AD transient / precomputed (112-127) ── */
    S_AD_IS_FORWARD=112,     /* indicator: executing AD forward op this cycle */
    S_AD_IS_BACKWARD=113,    /* indicator: in backward pass */
    S_AD_GRAD_ACCUM=114,     /* gradient accumulator */
    S_AD_UNARY_RELU_ACTIVE=S_AD_GRAD_ACCUM, /* forward scratch alias */
    S_AD_PROD_GRAD_LV=115,   /* precomputed: CUR_GRAD * RIGHT_VALUE (left delta for MUL) */
    S_AD_PROD_GRAD_RV=116,   /* precomputed: CUR_GRAD * LEFT_VALUE (right delta for MUL) */
    S_AD_LEFT_GRAD_NEW=117,  /* computed gradient delta for left parent */
    S_AD_RIGHT_GRAD_NEW=118, /* computed gradient delta for right parent */
    S_AD_PROD_LR=119,       /* precomputed: AD_LEFT_VALUE * AD_RIGHT_VALUE */
    S_AD_PROD_GRAD_CV=S_AD_PROD_LR,  /* legacy alias for dim 119 */
    S_AD_PROD_GRAD_SV=120,  /* precomputed: CUR_GRAD * CUR_SAVED (all unary backward) */
    S_AD_SPARE1=120,

    /* Stage-1 VM-as-transformer memory-op transients.
     * These reuse the true spare portion of Zone D. Layer 1 computes
     * saturated one-hot indicators over S_TYPE_TOS; Layer 3 consumes them
     * to execute NULL_P and the six type predicates without IS_NATIVE. */
    S_TYPE_IS_NUM=121,
    S_TYPE_IS_BOOL=122,
    S_TYPE_IS_PAIR=123,
    S_TYPE_IS_PROC=124,
    S_TYPE_IS_STR=125,
    S_TYPE_IS_VEC=126,
    S_TYPE_IS_NIL=127,

    S_AD_SPARE2=121, S_AD_SPARE3=122, S_AD_SPARE4=123,
    S_AD_SPARE5=124, S_AD_SPARE6=125, S_AD_SPARE7=126, S_AD_SPARE8=127,

    /* ── Zone E: bounded arena bank (128-207) ──
     * Cell i stores [kind, car_value, cdr_value, car_type, cdr_type].
     * Stack values hold small cell indices, not host pointers. */
    S_ARENA_BASE=128,
    S_ARENA_NEXT=S_ARENA_BASE + ARENA_CELLS * ARENA_CELL_FIELDS,

    /* Arena operation transients, cleared every cycle. */
    S_ARENA_WRITE_KIND,
    S_ARENA_WRITE_CAR,
    S_ARENA_WRITE_CDR,
    S_ARENA_READ_CAR,
    S_ARENA_READ_CDR,
    S_ARENA_TARGET,
    S_ARENA_NEW_KIND,
    S_ARENA_NEW_CAR,
    S_ARENA_NEW_CDR,
    S_ARENA_NEW_CAR_TYPE,
    S_ARENA_NEW_CDR_TYPE,
    S_ARENA_VEC_WRITE,
    S_ARENA_VEC_BASE,
    S_ARENA_VEC_LEN,
    S_ARENA_VEC_E0,
    S_ARENA_VEC_E1,
    S_ARENA_VEC_E2,
    S_ARENA_VEC_E3,
    S_ARENA_VEC_T0,
    S_ARENA_VEC_T1,
    S_ARENA_VEC_T2,
    S_ARENA_VEC_T3,
    S_ARENA_VEC_HAS_E0,
    S_ARENA_VEC_HAS_E1,
    S_ARENA_VEC_HAS_E2,
    S_ARENA_VEC_HAS_E3,
    S_ARENA_LIST_BASE,
    S_ARENA_LIST_E0,
    S_ARENA_LIST_E1,
    S_ARENA_LIST_E2,
    S_ARENA_LIST_E3,
    S_ARENA_LIST_T0,
    S_ARENA_LIST_T1,
    S_ARENA_LIST_T2,
    S_ARENA_LIST_T3,
    S_ARENA_LIST_CDR0,
    S_ARENA_LIST_CDR1,
    S_ARENA_LIST_CDR2,
    S_ARENA_LIST_CDR3,
    S_ARENA_LIST_CDRT0,
    S_ARENA_LIST_CDRT1,
    S_ARENA_LIST_CDRT2,
    S_ARENA_LIST_CDRT3,
    S_ARENA_LIST_HAS_E0,
    S_ARENA_LIST_HAS_E1,
    S_ARENA_LIST_HAS_E2,
    S_ARENA_LIST_HAS_E3,
    S_ARENA_TRANSIENT_START=S_ARENA_WRITE_KIND,
    S_ARENA_TRANSIENT_END=S_ARENA_LIST_HAS_E3
};

/* AD tape node field offsets within each 8-field block */
#define AD_F_OP    0
#define AD_F_VALUE 1
#define AD_F_GRAD  2
#define AD_F_LEFT  3
#define AD_F_RIGHT 4
#define AD_F_SAVED 5

/* Access tape node field: state[S_AD_TAPE_BASE + node * AD_NODE_FIELDS + field] */
#define AD_NODE(s, node, field) ((s)[S_AD_TAPE_BASE + (node) * AD_NODE_FIELDS + (field)])

/* Arena cell field offsets */
#define ARENA_F_KIND     0
#define ARENA_F_CAR_VAL  1
#define ARENA_F_CDR_VAL  2
#define ARENA_F_CAR_TYPE 3
#define ARENA_F_CDR_TYPE 4
#define ARENA_DIM(cell, field) (S_ARENA_BASE + (cell) * ARENA_CELL_FIELDS + (field))
#define ARENA_FIELD(s, cell, field) ((s)[ARENA_DIM((cell), (field))])

/* AD operation type encodings (stored in AD_F_OP field) */
#define AD_OP_CONST    0.0f
#define AD_OP_VAR      1.0f
#define AD_OP_ADD      2.0f
#define AD_OP_SUB      3.0f
#define AD_OP_MUL      4.0f
#define AD_OP_NEG      5.0f
#define AD_OP_ABS      6.0f
#define AD_OP_RELU     7.0f
#define AD_OP_SIGMOID  8.0f
#define AD_OP_TANH     9.0f
#define AD_OP_EXP     10.0f
#define AD_OP_LOG     11.0f
#define AD_OP_SQRT    12.0f
#define AD_OP_DIV     13.0f
#define AD_OP_POW     14.0f
#define AD_OP_SIN     15.0f
#define AD_OP_COS     16.0f

/* Bounded exact integer DIV/MOD artifact slice.
 * DIV is encoded for positive integer denominators 1..16 via denominator
 * gates and linear reciprocal weights. MOD is encoded as an exact lookup for
 * the positive numerator/denominator pairs exercised by the verifier range. */
#define DIV_WEIGHT_MAX_DENOM 16
#define MOD_WEIGHT_MAX_NUM   21
#define AD_POW_WEIGHT_MAX_BASE 8
#define AD_POW_WEIGHT_MAX_EXP  4

/* Type tag values */
#define TYPE_NUMBER  0.0f
#define TYPE_BOOL    1.0f
#define TYPE_PAIR    2.0f
#define TYPE_CLOSURE 3.0f
#define TYPE_STRING  4.0f
#define TYPE_VECTOR  5.0f
#define TYPE_NIL     6.0f
#define TYPE_CONT    7.0f

/*******************************************************************************
 * Utility Functions
 ******************************************************************************/

/** @brief Saturating sigmoid: clamps to exactly 0.0/1.0 outside +/-20 so the
 *         simulated-transformer gates agree bit-for-bit with the matrix
 *         forward pass (see SCALE comment above). */
static float sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/** @brief Soft equality gate implemented as the difference of two
 *         saturating sigmoids: ~1.0 when x == k, ~0.0 otherwise (used to
 *         simulate one-hot opcode/operand dispatch as a differentiable
 *         weight computation). */
static float indicator(float x, float k) {
    return sigmoidf(SCALE * (x - k + 0.5f)) - sigmoidf(SCALE * (x - k - 0.5f));
}

/** @brief Encode one program instruction as a Layer-0 embedding vector: a
 *         position key (linear + Gaussian-decay term for the attention
 *         softmax) plus the opcode/operand as value fields. */
static void embed_instruction(const Instr* instr, int position, float out[D]) {
    memset(out, 0, D * sizeof(float));
    out[0]  = (float)position;                        /* key[0] = position */
    out[1]  = -(float)(position * position) / 2.0f;   /* key[1] = -pos²/2 */
    out[S_OPCODE]  = (float)instr->op;                /* value[0] = opcode */
    out[S_OPERAND] = (float)instr->operand;           /* value[1] = operand */
}

/* Forward declarations for exec loop */
typedef struct {
    float return_pc;
    float saved_mem[MEM_SIZE];
    float saved_tos, saved_sos, saved_r2, saved_r3, saved_depth;
    float saved_closure;
} CallFrame;
static CallFrame g_frames[64];
static int g_frame_count = 0;
static void exec_loop_postprocess(float x[D], const Instr* prog, int n_instr);

/* Simple heap for CONS/CAR/CDR (pairs stored as consecutive float pairs).
 * WARNING: NOT thread-safe. All globals must be reset between program runs
 * via g_heap_ptr = 0. Do not use from multiple threads concurrently. */
#define HEAP_SIZE 4096
static float g_heap[HEAP_SIZE];
static int g_heap_ptr = 0;

/* Exception handler stack */
#define MAX_EXC_HANDLERS 32
static struct {
    float handler_pc;
    float saved_depth;
    float saved_mem[4]; /* MEM_SIZE */
    float saved_tos, saved_sos, saved_r2, saved_r3;
    float saved_type_tos, saved_type_sos, saved_type_r2, saved_type_r3;
} g_exc_handlers[MAX_EXC_HANDLERS];
static int g_exc_count = 0;
static float g_current_exn = 0.0f;

/* Closure tracking for GET_UPVALUE */
static int g_current_closure_ptr = -1;

/* Dynamic-wind stack */
#define MAX_WINDS 32
static struct {
    float after_thunk_ptr; /* heap index of after thunk closure */
    int frame_depth;
} g_wind_stack[MAX_WINDS];
static int g_wind_depth = 0;

/* Arena-based region stack for runtime libraries (complex, bignum, etc.) */
static VmRegionStack g_vm_regions;
static int g_vm_regions_initialized = 0;

/** @brief Lazily initialize and return the process-global arena region
 *         stack used by the runtime helper libraries (complex, bignum,
 *         etc.) linked into this interpreter. */
static VmRegionStack* vm_get_regions(void) {
    if (!g_vm_regions_initialized) {
        vm_region_stack_init(&g_vm_regions);
        g_vm_regions_initialized = 1;
    }
    return &g_vm_regions;
}

/*******************************************************************************
 * 1. Reference Interpreter (direct C, ground truth)
 ******************************************************************************/

typedef struct { float s[D]; } State;

/* ---- weight_matrices.c: thin aggregator over focused sub-files ----
#include "wm_reference.c"
#include "wm_layers.c"
#include "wm_exec.c"
#include "wm_simulated.c"
#include "wm_weights.c"
#include "wm_train.c"
#include "wm_gcheck.c"
#include "wm_main.c"
*/
