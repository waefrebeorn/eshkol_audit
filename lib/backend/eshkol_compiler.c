/**
 * @file eshkol_compiler.c
 * @brief Eshkol source → bytecode compiler targeting eshkol_vm.c ISA.
 *
 * Compiles S-expression source to the 38-opcode bytecode format.
 * Supports: arithmetic, comparisons, let/define, if/cond, do loops,
 * function definitions, lambda, closures, cons/car/cdr, display.
 *
 * Usage: ./eshkol_compiler [file.esk]
 *        Reads .esk, compiles to bytecode, executes via eshkol_vm.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>

/* ESKB binary format writer (single-file include pattern) */
#include "eskb_writer.c"

/* ESKB binary format reader (single-file include pattern) */
#include "eskb_reader.c"

/* Opcodes (must match eshkol_vm.c) */
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
    /* Vectors */
    OP_VEC_CREATE=39,   /* operand = count; pops count values, creates vector */
    OP_VEC_REF=40,      /* TOS=index, SOS=vector → push vector[index] */
    OP_VEC_SET=41,      /* TOS=value, SOS=index, TOS-2=vector → set */
    OP_VEC_LEN=42,      /* TOS=vector → push length */
    /* Strings */
    OP_STR_REF=43,      /* TOS=index, SOS=string → push char */
    OP_STR_LEN=44,      /* TOS=string → push length */
    /* Type checks */
    OP_PAIR_P=45,       /* TOS → push (pair? TOS) */
    OP_NUM_P=46,        /* TOS → push (number? TOS) */
    OP_STR_P=47,        /* TOS → push (string? TOS) */
    OP_BOOL_P=48,       /* TOS → push (boolean? TOS) */
    OP_PROC_P=49,       /* TOS → push (procedure? TOS) */
    OP_VEC_P=50,        /* TOS → push (vector? TOS) */
    /* Set mutations */
    OP_SET_CAR=51,      /* TOS=val, SOS=pair → set car */
    OP_SET_CDR=52,      /* TOS=val, SOS=pair → set cdr */
    OP_POPN=53,         /* operand=N: pop N values below TOS, keeping TOS (scope cleanup) */
    OP_OPEN_CLOSURE=54,
    OP_CALLCC=55,       /* call/cc: capture continuation, call TOS with it */
    OP_INVOKE_CC=56,    /* invoke a captured continuation with a value */
    OP_PUSH_HANDLER=57, /* operand=handler_pc: save continuation, push exception handler */
    OP_POP_HANDLER=58,  /* remove topmost exception handler (normal guard exit) */
    OP_GET_EXN=59,      /* push current exception value (set by raise) */
    OP_PACK_REST=60,    /* operand=n_fixed: pack args from fp+n_fixed..sp into list at fp+n_fixed */
    OP_WIND_PUSH=61,    /* push after thunk onto wind stack */
    OP_WIND_POP=62,     /* pop from wind stack */

    OP_COUNT=63
} OpCode;

typedef struct { uint8_t op; int32_t operand; } Instr;

/* Value types for constant pool */
typedef enum {
    VAL_NIL=0, VAL_INT=1, VAL_FLOAT=2, VAL_BOOL=3,
    VAL_PAIR=4, VAL_CLOSURE=5, VAL_STRING=6, VAL_VECTOR=7,
    VAL_CONTINUATION=8, VAL_HASH=9
} ValType;
typedef struct { ValType type; union { int64_t i; double f; int b; int32_t ptr; } as; } Value;
#define INT_VAL(v) ((Value){.type=VAL_INT, .as.i=(v)})
#define FLOAT_VAL(v) ((Value){.type=VAL_FLOAT, .as.f=(v)})

/*******************************************************************************
 * S-Expression Parser (reused from stackvm_codegen.c)
 ******************************************************************************/

typedef enum { N_NUMBER, N_SYMBOL, N_LIST, N_STRING, N_BOOL } NodeType;
typedef struct Node {
    NodeType type;
    double numval;
    char symbol[128];
    struct Node** children;
    int n_children;
} Node;

/* Hygienic macro expansion (syntax-rules).
 * Define VM_MACRO_NODE_DEFINED to skip MacroNode's duplicate enum/struct.
 * Provide typedefs so vm_macro.c functions can use MacroNode/MacroNodeType
 * while actually operating on the compiler's Node type (layout-compatible). */
#define VM_MACRO_NODE_DEFINED
typedef NodeType MacroNodeType;
typedef struct MacroNode {
    MacroNodeType    type;
    double           numval;
    char             symbol[128];
    struct MacroNode** children;
    int              n_children;
    int              _cap;
} MacroNode;
#include "vm_macro.c"

static const char* src_ptr = NULL;
static int g_trace_on = 0;  /* global, set by --trace flag */

/** @brief Advance src_ptr past whitespace and `;`-to-end-of-line comments. */
static void skip_ws(void) {
    while (*src_ptr) {
        if (isspace(*src_ptr)) { src_ptr++; continue; }
        if (*src_ptr == ';') { while (*src_ptr && *src_ptr != '\n') src_ptr++; continue; }
        break;
    }
}

/** @brief Allocate and zero-initialize a new AST Node of type @p t. */
static Node* make_node(NodeType t) {
    Node* n = (Node*)calloc(1, sizeof(Node));
    if (!n) { fprintf(stderr, "ERROR: allocation failed in make_node\n"); return NULL; }
    n->type = t;
    return n;
}
/** @brief Append child @p c to parent Node @p p's children array, growing
 *         it via realloc. */
static void add_child(Node* p, Node* c) {
    if (!p || !c) return;
    Node** nc = (Node**)realloc(p->children, (p->n_children+1)*sizeof(Node*));
    if (!nc) { fprintf(stderr, "ERROR: allocation failed in add_child\n"); return; }
    p->children = nc;
    p->children[p->n_children++] = c;
}

static void free_node(Node* n);
static Node* parse_sexp(void);
/** @brief Parse a parenthesized list body (after the opening `(` has been
 *         consumed) into an N_LIST node, reading sub-expressions until `)`
 *         or end of input. */
static Node* parse_list(void) {
    Node* list = make_node(N_LIST);
    if (!list) return NULL;
    while (1) { skip_ws(); if (!*src_ptr || *src_ptr == ')') break; Node* c = parse_sexp(); if (!c) break; add_child(list, c); }
    if (*src_ptr == ')') src_ptr++;
    return list;
}

/**
 * @brief Recursive-descent S-expression reader: parses one datum from the
 *        global src_ptr cursor — lists, quote/quasiquote/unquote(-splicing)
 *        sugar, string literals (with \\n \\t \\\\ \\" escapes), #t/#f,
 *        character literals (#\\a, #\\space, #\\newline, #\\tab, #\\nul),
 *        vector literals #(...), numbers, and symbols.
 * @return The parsed Node, or NULL at end of input / on a bare `)`.
 */
static Node* parse_sexp(void) {
    skip_ws();
    if (!*src_ptr) return NULL;
    if (*src_ptr == '(') { src_ptr++; return parse_list(); }
    if (*src_ptr == ')') return NULL;
    if (*src_ptr == '\'') {
        src_ptr++;
        Node* q = make_node(N_LIST); if (!q) return NULL;
        Node* qs = make_node(N_SYMBOL); if (!qs) { free_node(q); return NULL; }
        strcpy(qs->symbol, "quote");
        add_child(q, qs);
        Node* datum = parse_sexp();
        if (datum) add_child(q, datum);
        return q;
    }
    /* Quasiquote */
    if (*src_ptr == '`') {
        src_ptr++;
        Node* q = make_node(N_LIST); if (!q) return NULL;
        Node* tag = make_node(N_SYMBOL); if (!tag) { free_node(q); return NULL; }
        strcpy(tag->symbol, "quasiquote");
        add_child(q, tag);
        Node* datum = parse_sexp();
        if (datum) add_child(q, datum);
        return q;
    }
    /* Unquote-splicing (must check before unquote) */
    if (*src_ptr == ',' && src_ptr[1] == '@') {
        src_ptr += 2;
        Node* q = make_node(N_LIST); if (!q) return NULL;
        Node* tag = make_node(N_SYMBOL); if (!tag) { free_node(q); return NULL; }
        strcpy(tag->symbol, "unquote-splicing");
        add_child(q, tag);
        Node* datum = parse_sexp();
        if (datum) add_child(q, datum);
        return q;
    }
    /* Unquote */
    if (*src_ptr == ',') {
        src_ptr++;
        Node* q = make_node(N_LIST); if (!q) return NULL;
        Node* tag = make_node(N_SYMBOL); if (!tag) { free_node(q); return NULL; }
        strcpy(tag->symbol, "unquote");
        add_child(q, tag);
        Node* datum = parse_sexp();
        if (datum) add_child(q, datum);
        return q;
    }
    /* String literal */
    if (*src_ptr == '"') {
        src_ptr++; /* skip opening quote */
        char buf[256]; int i = 0;
        while (*src_ptr && *src_ptr != '"' && i < 255) {
            if (*src_ptr == '\\' && src_ptr[1]) {
                src_ptr++;
                switch (*src_ptr) {
                    case 'n': buf[i++] = '\n'; break;
                    case 't': buf[i++] = '\t'; break;
                    case '\\': buf[i++] = '\\'; break;
                    case '"': buf[i++] = '"'; break;
                    default: buf[i++] = *src_ptr; break;
                }
                src_ptr++;
            } else {
                buf[i++] = *src_ptr++;
            }
        }
        if (*src_ptr == '"') src_ptr++; /* skip closing quote */
        buf[i] = 0;
        Node* n = make_node(N_STRING); if (!n) return NULL;
        strncpy(n->symbol, buf, 127); n->symbol[127] = 0;
        return n;
    }
    if (*src_ptr == '#') {
        if (src_ptr[1] == 't' && (src_ptr[2] == 0 || isspace(src_ptr[2]) || src_ptr[2] == ')')) {
            src_ptr += 2; Node* n = make_node(N_BOOL); if (!n) return NULL; n->numval = 1; strcpy(n->symbol, "#t"); return n;
        }
        if (src_ptr[1] == 'f' && (src_ptr[2] == 0 || isspace(src_ptr[2]) || src_ptr[2] == ')')) {
            src_ptr += 2; Node* n = make_node(N_BOOL); if (!n) return NULL; n->numval = 0; strcpy(n->symbol, "#f"); return n;
        }
        /* Character literal: #\a, #\space, #\newline, #\tab */
        if (src_ptr[1] == '\\') {
            src_ptr += 2;
            int ch;
            if (strncmp(src_ptr, "space", 5) == 0 && (!src_ptr[5] || isspace(src_ptr[5]) || src_ptr[5] == ')')) {
                ch = ' '; src_ptr += 5;
            } else if (strncmp(src_ptr, "newline", 7) == 0 && (!src_ptr[7] || isspace(src_ptr[7]) || src_ptr[7] == ')')) {
                ch = '\n'; src_ptr += 7;
            } else if (strncmp(src_ptr, "tab", 3) == 0 && (!src_ptr[3] || isspace(src_ptr[3]) || src_ptr[3] == ')')) {
                ch = '\t'; src_ptr += 3;
            } else if (strncmp(src_ptr, "nul", 3) == 0 && (!src_ptr[3] || isspace(src_ptr[3]) || src_ptr[3] == ')')) {
                ch = 0; src_ptr += 3;
            } else {
                ch = (unsigned char)*src_ptr; src_ptr++;
            }
            Node* n = make_node(N_NUMBER); if (!n) return NULL;
            n->numval = ch;
            return n;
        }
        /* Vector literal: #(elements...) */
        if (src_ptr[1] == '(') {
            src_ptr += 2; /* skip #( */
            Node* vec = make_node(N_LIST); if (!vec) return NULL;
            Node* tag = make_node(N_SYMBOL); if (!tag) { free_node(vec); return NULL; }
            strcpy(tag->symbol, "vector");
            add_child(vec, tag);
            while (1) { skip_ws(); if (!*src_ptr || *src_ptr == ')') break; Node* el = parse_sexp(); if (!el) break; add_child(vec, el); }
            if (*src_ptr == ')') src_ptr++;
            return vec;
        }
    }
    /* Number */
    if (isdigit(*src_ptr) || (*src_ptr == '-' && isdigit(src_ptr[1]))) {
        char buf[64]; int i = 0;
        if (*src_ptr == '-') buf[i++] = *src_ptr++;
        while ((isdigit(*src_ptr) || *src_ptr == '.') && i < 63) buf[i++] = *src_ptr++;
        buf[i] = 0;
        Node* n = make_node(N_NUMBER); if (!n) return NULL; n->numval = atof(buf); return n;
    }
    /* Symbol */
    char buf[128]; int i = 0;
    while (*src_ptr && !isspace(*src_ptr) && *src_ptr != '(' && *src_ptr != ')' && *src_ptr != '"' && i < 127)
        buf[i++] = *src_ptr++;
    buf[i] = 0;
    Node* n = make_node(N_SYMBOL); if (!n) return NULL; strncpy(n->symbol, buf, 127); n->symbol[127] = 0; return n;
}

/** @brief Recursively free an AST Node and all its children. */
static void free_node(Node* n) { if (!n) return; for (int i=0;i<n->n_children;i++) free_node(n->children[i]); free(n->children); free(n); }

/*******************************************************************************
 * Compiler: AST → Bytecode
 ******************************************************************************/

#define MAX_CODE 32768
#define MAX_CONSTS 1024
#define MAX_LOCALS 512
#define MAX_FUNCS 64

typedef struct {
    char name[128];
    int slot;
    int depth;
    int boxed;  /* 1 = variable is heap-boxed (stored in 1-element vector) */
} Local;

typedef struct {
    char name[128];
    int enclosing_slot;  /* slot or upvalue index in the enclosing scope */
    int index;           /* upvalue index in this closure */
    int is_local;        /* 1 = enclosing_slot is a local, 0 = it's an upvalue */
    int boxed;           /* 1 = the captured variable is heap-boxed */
} Upvalue;

#define MAX_UPVALUES 32

typedef struct FuncChunk {
    Instr code[MAX_CODE];
    int code_len;
    Value constants[MAX_CONSTS];
    int n_constants;
    Local locals[MAX_LOCALS];
    int n_locals;
    Upvalue upvalues[MAX_UPVALUES];
    int n_upvalues;
    int scope_depth;
    int scope_stack_base[32]; /* stack depth at scope entry, for cleanup on exit */
    struct FuncChunk* enclosing;
    int param_count;
    int stack_depth;  /* compile-time stack depth (values above fp) */
} FuncChunk;

/** @brief Check whether Node @p n is a symbol equal to string @p s. */
static int is_sym(Node* n, const char* s) { return n && n->type == N_SYMBOL && strcmp(n->symbol, s) == 0; }

/** @brief Append one bytecode instruction (@p op, @p operand) to chunk
 *         @p c, erroring if MAX_CODE is exceeded. */

/* ---- eshkol_compiler.c: thin aggregator over focused sub-files ----
#include "ec_front.c"
#include "ec_compile.c"
#include "ec_print.c"
#include "ec_peephole.c"
#include "ec_exec.c"
#include "ec_main.c"
*/
