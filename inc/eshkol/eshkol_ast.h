#ifndef ESHKOL_AST_H
#define ESHKOL_AST_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "eshkol_value.h"
#include "eshkol_type.h"
#include "eshkol_macro.h"
#include "eshkol_hott.h"

#ifdef __cplusplus
#include <fstream>
#include <istream>
extern "C" {
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ESHKOL_INVALID_OP,
    ESHKOL_COMPOSE_OP,
    ESHKOL_IF_OP,
    ESHKOL_ADD_OP,
    ESHKOL_SUB_OP,
    ESHKOL_MUL_OP,
    ESHKOL_DIV_OP,
    ESHKOL_CALL_OP,
    ESHKOL_DEFINE_OP,
    ESHKOL_SEQUENCE_OP,
    ESHKOL_EXTERN_OP,
    ESHKOL_EXTERN_VAR_OP,
    ESHKOL_LAMBDA_OP,
    ESHKOL_LET_OP,
    ESHKOL_LET_STAR_OP,  // let* - sequential bindings
    ESHKOL_LETREC_OP,    // letrec - recursive bindings (all bindings visible to all values)
    ESHKOL_LETREC_STAR_OP, // letrec* - sequential recursive bindings (R7RS: left-to-right evaluation)
    ESHKOL_AND_OP,       // short-circuit and
    ESHKOL_OR_OP,        // short-circuit or
    ESHKOL_COND_OP,      // multi-branch conditional
    ESHKOL_CASE_OP,      // case expression (switch on value)
    ESHKOL_MATCH_OP,     // pattern matching (match expr (pattern body) ...)
    ESHKOL_DO_OP,        // do loop (iteration construct)
    ESHKOL_WHEN_OP,      // when - one-armed if (execute when true)
    ESHKOL_UNLESS_OP,    // unless - negated when (execute when false)
    ESHKOL_QUOTE_OP,     // quote - literal data
    ESHKOL_QUASIQUOTE_OP,        // quasiquote (`) - template with unquotes
    ESHKOL_UNQUOTE_OP,           // unquote (,) - escape from quasiquote
    ESHKOL_UNQUOTE_SPLICING_OP,  // unquote-splicing (,@) - splice list into quasiquote
    ESHKOL_SET_OP,       // set! - variable mutation
    ESHKOL_DEFINE_TYPE_OP, // define-type - type alias definition
    ESHKOL_IMPORT_OP,    // import - load another Eshkol file (legacy string path)
    ESHKOL_REQUIRE_OP,   // require - import module by symbolic name (new module system)
    ESHKOL_PROVIDE_OP,   // provide - export symbols from module
    // Memory management operators (OALR - Ownership-Aware Lexical Regions)
    ESHKOL_WITH_REGION_OP,  // with-region - lexical region for batch allocation/free
    ESHKOL_OWNED_OP,        // owned - linear type for resources
    ESHKOL_MOVE_OP,         // move - transfer ownership
    ESHKOL_BORROW_OP,       // borrow - temporary read-only access
    ESHKOL_SHARED_OP,       // shared - reference-counted allocation
    ESHKOL_WEAK_REF_OP,     // weak-ref - weak reference (doesn't prevent cleanup)
    ESHKOL_TENSOR_OP,
    ESHKOL_DIFF_OP,
    // Automatic differentiation operators
    ESHKOL_DERIVATIVE_OP,
    ESHKOL_GRADIENT_OP,
    ESHKOL_JACOBIAN_OP,
    ESHKOL_HESSIAN_OP,
    ESHKOL_DIVERGENCE_OP,
    ESHKOL_CURL_OP,
    ESHKOL_LAPLACIAN_OP,
    ESHKOL_DIRECTIONAL_DERIV_OP,
    ESHKOL_TAYLOR_OP,           // (taylor f x k) - K+1 Taylor coefficients (ESH-0186)
    ESHKOL_DERIVATIVE_N_OP,     // (derivative-n f x k) - arbitrary-order n-th derivative (ESH-0186)
    // HoTT Type System operators
    ESHKOL_TYPE_ANNOTATION_OP,  // (: name type) - standalone type declaration
    ESHKOL_FORALL_OP,           // (forall (a b) type) - polymorphic type
    // Exception handling operators
    ESHKOL_GUARD_OP,            // (guard (var clause ...) body ...) - exception handler
    ESHKOL_RAISE_OP,            // (raise exception) - raise exception
    // Multiple return values operators
    ESHKOL_LET_VALUES_OP,       // (let-values (((vars ...) producer) ...) body)
    ESHKOL_LET_STAR_VALUES_OP,  // (let*-values (((vars ...) producer) ...) body) - sequential
    ESHKOL_VALUES_OP,           // (values v1 v2 ...) - return multiple values
    ESHKOL_CALL_WITH_VALUES_OP, // (call-with-values producer consumer)
    // Macro system operators
    ESHKOL_DEFINE_SYNTAX_OP,    // (define-syntax name (syntax-rules ...))
    ESHKOL_LET_SYNTAX_OP,      // (let-syntax ((name (syntax-rules ...)) ...) body ...)
    ESHKOL_LETREC_SYNTAX_OP,   // (letrec-syntax ((name (syntax-rules ...)) ...) body ...)
    // First-class continuations
    ESHKOL_CALL_CC_OP,         // (call/cc proc) or (call-with-current-continuation proc)
    ESHKOL_DYNAMIC_WIND_OP,    // (dynamic-wind before thunk after)
    // Neuro-symbolic consciousness engine operations
    ESHKOL_LOGIC_VAR_OP,              // ?x - create/reference logic variable
    ESHKOL_UNIFY_OP,                  // (unify t1 t2 subst) -> subst|#f
    ESHKOL_MAKE_SUBST_OP,             // (make-substitution) -> empty subst
    ESHKOL_WALK_OP,                   // (walk term subst) -> resolved term
    ESHKOL_MAKE_FACT_OP,              // (make-fact 'pred arg...) -> fact
    ESHKOL_MAKE_KB_OP,                // (make-kb) -> empty KB
    ESHKOL_KB_ASSERT_OP,              // (kb-assert! kb fact) -> void
    ESHKOL_KB_QUERY_OP,               // (kb-query kb pattern) -> list of substs
    ESHKOL_MAKE_FACTOR_GRAPH_OP,      // (make-factor-graph n-vars dims) -> fg
    ESHKOL_FG_ADD_FACTOR_OP,          // (fg-add-factor! fg vars cpt) -> void
    ESHKOL_FG_INFER_OP,               // (fg-infer! fg iterations) -> beliefs
    ESHKOL_FREE_ENERGY_OP,            // (free-energy beliefs log-joint) -> scalar
    ESHKOL_EXPECTED_FREE_ENERGY_OP,   // (efe model action-var action-state) -> scalar
    ESHKOL_MAKE_WORKSPACE_OP,         // (make-workspace dim max-modules) -> ws
    ESHKOL_WS_REGISTER_OP,            // (ws-register! ws name module) -> void
    ESHKOL_WS_STEP_OP,                // (ws-step! ws) -> broadcast-content
    ESHKOL_FG_UPDATE_CPT_OP,          // (fg-update-cpt! fg factor-idx new-cpt) -> fg
    ESHKOL_FG_OBSERVE_OP,             // (fg-observe! fg var-id observed-state) -> bool
    ESHKOL_LOGIC_VAR_PRED_OP,         // (logic-var? x) -> bool
    ESHKOL_SUBSTITUTION_PRED_OP,      // (substitution? x) -> bool
    ESHKOL_KB_PRED_OP,                // (kb? x) -> bool
    ESHKOL_FACT_PRED_OP,              // (fact? x) -> bool
    ESHKOL_FACTOR_GRAPH_PRED_OP,      // (factor-graph? x) -> bool
    ESHKOL_WORKSPACE_PRED_OP,         // (workspace? x) -> bool
    // ===== R7RS WAVE 3 SPECIAL FORMS =====
    ESHKOL_CASE_LAMBDA_OP,           // (case-lambda ((formals) body ...) ...) - transformed at parse time
    ESHKOL_DEFINE_RECORD_TYPE_OP,    // (define-record-type name ctor pred field ...) - transformed at parse time
    ESHKOL_PARAMETERIZE_OP,          // (parameterize ((param val) ...) body ...) - transformed at parse time
    ESHKOL_MAKE_PARAMETER_OP,        // (make-parameter init) - transformed at parse time
    ESHKOL_COND_EXPAND_OP,           // (cond-expand (feature body ...) ...) - transformed at parse time
    ESHKOL_INCLUDE_OP,               // (include "file" ...) - transformed at parse time
    ESHKOL_SYNTAX_ERROR_OP,          // (syntax-error "msg" datum ...) - handled at parse time
    ESHKOL_KB_QUERY_PREFIX_OP,       // (kb-query-prefix kb pattern) -> list of substs (pattern arity ≤ fact arity)
    // ===== Differentiable external memory (core.dnc) — appended to preserve ABI =====
    ESHKOL_DNC_MAKE_OP,              // (make-dnc-memory N W) -> mem opaque handle
    ESHKOL_DNC_CONTENT_ADDR_OP,     // (dnc-content-address mem key beta) -> length-N wvec
    ESHKOL_DNC_LOC_ADDR_OP,         // (dnc-loc-address addr beta N) -> length-N wvec
    ESHKOL_DNC_READ_OP,             // (dnc-read mem wvec) -> length-W row
    ESHKOL_DNC_WRITE_OP,            // (dnc-write! mem wvec erase add) -> mem
    ESHKOL_DNC_ALLOC_WEIGHTS_OP,    // (dnc-alloc-weights mem beta) -> length-N wvec
    ESHKOL_DNC_READ_GRAD_OP,        // (dnc-read-grad mem key target beta) -> (dkey . dmem)
    ESHKOL_DNC_PRED_OP,             // (dnc-memory? x) -> bool
    // ===== SDNC weight-program θ (core.sdnc) — appended to preserve ABI =====
    ESHKOL_SDNC_PROGRAM_OP,         // (sdnc-program name) -> θ opaque handle
    ESHKOL_SDNC_RUN_OP,             // (sdnc-run θ input) -> output vector
    ESHKOL_SDNC_WEIGHT_GRAD_OP,     // (sdnc-weight-grad θ input target) -> flat ∂L/∂weights
    ESHKOL_SDNC_PARAMS_OP,          // (sdnc-params θ) -> vector of flattened trainable weights
    ESHKOL_SDNC_SET_PARAMS_OP,      // (sdnc-set-params! θ vec) -> θ
    ESHKOL_SDNC_IMPROVE_OP,         // (sdnc-improve! θ data steps lr) -> θ
    ESHKOL_SDNC_PRED_OP,            // (sdnc? x) -> bool
} eshkol_op_t;

struct eshkol_ast;
struct eshkol_operation;

/**
 * @brief Tagged union of operands for every special form/operator kind.
 *
 * `op` (eshkol_op_t) selects which anonymous struct in the union is active;
 * each sub-struct holds the AST subtrees and scalar parameters specific to
 * that form (e.g. `if_op` holds true/false branches, `lambda_op` holds
 * parameters/body/captures, `derivative_op` holds the function/point/mode
 * for automatic differentiation). This is the payload referenced by
 * eshkol_ast_t::operation for AST nodes representing operations rather
 * than literals or variable references.
 */
typedef struct eshkol_operation {
    eshkol_op_t op;
    union {
        struct {
            struct eshkol_ast *base;
            struct eshkol_ast *ptr;
        } assign_op;
        struct {
            struct eshkol_ast *func_a;
            struct eshkol_ast *func_b;
        } compose_op;
        struct {
            struct eshkol_operation *if_true;
            struct eshkol_operation *if_false;
        } if_op;
        struct {
            struct eshkol_ast *func;
            struct eshkol_ast *variables;
            uint64_t num_vars;
        } call_op;
        struct {
            char *name;
            struct eshkol_ast *value;
            uint8_t is_function;
            struct eshkol_ast *parameters;
            uint64_t num_params;
            uint8_t is_variadic;      // True if function accepts variable arguments
            char *rest_param;         // Name of rest parameter (for variadic functions)
            uint8_t is_external;      // True if function is external (body from linked .o)
            // HoTT type annotations
            hott_type_expr_t *return_type;    // Return type annotation (NULL if not annotated)
            hott_type_expr_t **param_types;   // Array of parameter type annotations (NULL entries for unannotated)
            char *link_section;       // Optional section name for top-level symbols
            uint64_t alignment;       // Optional ABI alignment
            uint8_t has_alignment;    // True if alignment was explicitly set
            uint8_t is_used;          // Force retention through llvm.used
            uint8_t is_weak;          // Emit weak linkage
            uint8_t export_symbol;    // Force public/exported linkage for the definition
            char *export_name;        // Optional emitted symbol name
            uint8_t is_no_return;     // Function never returns normally
        } define_op;
        struct {
            struct eshkol_ast *expressions;
            uint64_t num_expressions;
        } sequence_op;
        struct {
            char *name;
            char *real_name;
            char *return_type;
            struct eshkol_ast *parameters;
            uint64_t num_params;
            uint8_t is_weak;          // Imported function may be absent at link/load time
            uint8_t is_no_return;     // Imported function never returns normally
        } extern_op;
        struct {
            char *name;
            char *type;
            char *real_name;
        } extern_var_op;
	struct {
	           struct eshkol_ast *parameters;
	           uint64_t num_params;
	           struct eshkol_ast *body;
	           struct eshkol_ast *captured_vars;
	           uint64_t num_captured;
	           uint8_t is_variadic;       // True if lambda accepts variable arguments
	           char *rest_param;          // Name of rest parameter (for variadic lambdas)
	           // HoTT type annotations
	           hott_type_expr_t *return_type;    // Return type annotation (NULL if not annotated)
	           hott_type_expr_t **param_types;   // Array of parameter type annotations (NULL entries for unannotated)
	       } lambda_op;
	       struct {
	           struct eshkol_ast *bindings;      // Array of (variable value) pairs
	           uint64_t num_bindings;
	           struct eshkol_ast *body;
	           char *name;                       // Named let: loop name (NULL for regular let)
	           // HoTT type annotations for bindings
	           hott_type_expr_t **binding_types; // Array of type annotations (NULL entries for unannotated)
	       } let_op;
	       struct {
	           char *name;                       // Variable name to mutate
	           struct eshkol_ast *value;         // New value
	       } set_op;
	       struct {
	           char *name;                       // Type alias name
	           hott_type_expr_t *type_expr;      // The type expression this aliases
	           char **type_params;               // Type parameter names (for parameterized types)
	           uint64_t num_type_params;         // Number of type parameters
	       } define_type_op;
	       struct {
	           char *path;                       // Path to file to import
	       } import_op;
	       struct {
	           char **module_names;              // Array of symbolic module names (e.g., "data.json")
	           uint64_t num_modules;             // Number of modules to require
	           char **import_prefixes;           // Optional R7RS bare-prefix aliases per module
	           char ***import_except_names;       // Optional R7RS except lists per module
	           uint64_t *num_import_except_names; // Lengths for import_except_names entries
	       } require_op;
	       struct {
	           char **export_names;              // Array of exported symbol names
	           uint64_t num_exports;             // Number of symbols to export
	       } provide_op;
	       // ===== MEMORY MANAGEMENT OPERATIONS (OALR) =====
	       struct {
	           char *name;                       // Optional region name (NULL for anonymous)
	           uint64_t size_hint;               // Optional size hint in bytes (0 for default)
	           struct eshkol_ast *body;          // Body expressions to execute in region
	           uint64_t num_body_exprs;          // Number of body expressions
	       } with_region_op;
	       struct {
	           struct eshkol_ast *value;         // Value to mark as owned
	       } owned_op;
	       struct {
	           struct eshkol_ast *value;         // Value to transfer ownership of
	       } move_op;
	       struct {
	           struct eshkol_ast *value;         // Value to borrow
	           struct eshkol_ast *body;          // Body expressions during borrow
	           uint64_t num_body_exprs;          // Number of body expressions
	       } borrow_op;
	       struct {
	           struct eshkol_ast *value;         // Value to make shared (ref-counted)
	       } shared_op;
	       struct {
	           struct eshkol_ast *value;         // Shared value to create weak ref from
	       } weak_ref_op;
	       // ===== END MEMORY MANAGEMENT OPERATIONS =====
	       struct {
            struct eshkol_ast *elements;
            uint64_t *dimensions;
            uint64_t num_dimensions;
            uint64_t total_elements;
        } tensor_op;
        struct {
            struct eshkol_ast *expression;  // Expression to differentiate
            char *variable;                 // Variable to differentiate with respect to
        } diff_op;
        struct {
            struct eshkol_ast *function;    // Function to differentiate (lambda or function reference)
            struct eshkol_ast *point;       // Point to evaluate derivative at
            uint8_t mode;                   // 0=forward, 1=reverse, 2=auto (for future use)
        } derivative_op;
        struct {
            struct eshkol_ast *function;    // Function to differentiate (univariate)
            struct eshkol_ast *point;       // Point x0 to evaluate at
            struct eshkol_ast *order;       // Requested order k (may be a runtime value)
        } taylor_op;                        // shared by ESHKOL_TAYLOR_OP / ESHKOL_DERIVATIVE_N_OP
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate gradient at
        } gradient_op;
        struct {
            struct eshkol_ast *function;    // Vector field function: ℝⁿ → ℝᵐ
            struct eshkol_ast *point;       // Point to evaluate jacobian at
        } jacobian_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate hessian at
        } hessian_op;
        struct {
            struct eshkol_ast *function;    // Vector field function: ℝⁿ → ℝⁿ
            struct eshkol_ast *point;       // Point to evaluate divergence at
        } divergence_op;
        struct {
            struct eshkol_ast *function;    // Vector field function: ℝ³ → ℝ³
            struct eshkol_ast *point;       // Point to evaluate curl at
        } curl_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate laplacian at
        } laplacian_op;
        struct {
            struct eshkol_ast *function;    // Scalar field function: ℝⁿ → ℝ
            struct eshkol_ast *point;       // Point to evaluate directional derivative at
            struct eshkol_ast *direction;   // Direction vector
        } directional_deriv_op;
        // ===== HoTT Type System Operations =====
        struct {
            char *name;                     // Name being annotated
            hott_type_expr_t *type_expr;    // The type expression
        } type_annotation_op;
        struct {
            char **type_vars;               // Quantified type variable names
            uint64_t num_vars;
            hott_type_expr_t *body;         // Body type expression
        } forall_op;
        // ===== EXCEPTION HANDLING OPERATIONS =====
        struct {
            char *var_name;                 // Exception variable name (bound in clauses)
            struct eshkol_ast *clauses;     // Array of guard clauses: ((test expr ...) ...)
            uint64_t num_clauses;           // Number of clauses
            struct eshkol_ast *body;        // Body expressions to evaluate
            uint64_t num_body_exprs;        // Number of body expressions
        } guard_op;
        struct {
            struct eshkol_ast *exception;   // Exception value to raise
        } raise_op;
        // ===== END EXCEPTION HANDLING OPERATIONS =====
        // ===== MULTIPLE RETURN VALUES OPERATIONS =====
        struct {
            struct eshkol_ast *expressions; // Array of expressions to return
            uint64_t num_values;            // Number of values to return
        } values_op;
        struct {
            struct eshkol_ast *producer;    // Thunk that produces multiple values
            struct eshkol_ast *consumer;    // Function that consumes multiple values
        } call_with_values_op;
        struct {
            // Each binding is: ((var1 var2 ...) producer)
            // Stored as array of structs containing var names and producer
            char ***binding_vars;           // Array of arrays of variable names
            uint64_t *binding_var_counts;   // Count of vars per binding
            struct eshkol_ast *producers;   // Array of producer expressions
            uint64_t num_bindings;          // Number of bindings
            struct eshkol_ast *body;        // Body expression
        } let_values_op;
        // ===== END MULTIPLE RETURN VALUES OPERATIONS =====
        // ===== PATTERN MATCHING OPERATIONS =====
        struct {
            struct eshkol_ast *expr;             // Expression to match against
            eshkol_match_clause_t *clauses;      // Array of match clauses
            uint64_t num_clauses;                // Number of clauses
        } match_op;
        // ===== END PATTERN MATCHING OPERATIONS =====

        // ===== MACRO OPERATIONS =====
        struct {
            eshkol_macro_def_t *macro;           // Macro definition
        } define_syntax_op;
        struct {
            eshkol_macro_def_t **macros;         // Array of macro definitions
            uint64_t num_macros;                 // Number of macro bindings
            struct eshkol_ast *body;             // Body expression
        } let_syntax_op;
        // ===== END MACRO OPERATIONS =====
        // ===== CONTINUATION OPERATIONS =====
        struct {
            struct eshkol_ast *proc;             // Procedure to call with continuation
        } call_cc_op;
        struct {
            struct eshkol_ast *before;           // Before thunk
            struct eshkol_ast *thunk;            // Body thunk
            struct eshkol_ast *after;            // After thunk
        } dynamic_wind_op;
        // ===== END CONTINUATION OPERATIONS =====
        // ===== NEURO-SYMBOLIC CONSCIOUSNESS ENGINE OPERATIONS =====
        struct {
            uint64_t var_id;                     // Logic variable ID (global registry)
            const char *name;                    // Logic variable name (e.g., "?x")
        } logic_var_op;
        // ===== R7RS WAVE 3 OPERATIONS =====
        struct {
            // Each clause: (formals body ...)
            // formals stored as lambda_op sub-ASTs
            struct eshkol_ast *clauses;          // Array of lambda ASTs (one per clause)
            uint64_t num_clauses;                // Number of arity clauses
        } case_lambda_op;
        struct {
            // (parameterize ((param1 val1) (param2 val2) ...) body ...)
            struct eshkol_ast *params;           // Array of parameter expressions
            struct eshkol_ast *values;           // Array of value expressions
            uint64_t num_bindings;               // Number of parameter bindings
            struct eshkol_ast *body;             // Body expression
        } parameterize_op;
        // ===== END R7RS WAVE 3 OPERATIONS =====
    };
} eshkol_operations_t;

/**
 * @brief Frontend abstract-syntax-tree node.
 *
 * The single node type used throughout parsing, macro expansion, type
 * checking, and codegen. `type` (eshkol_type_t) selects the active union
 * member: scalar literal fields (`int64_val`, `double_val`, ...), string
 * literal storage (`str_val`), a function/lambda definition
 * (`eshkol_func`), a variable reference (`variable`), a cons cell
 * (`cons_cell`), a tensor literal (`tensor_val`), or a compound operation
 * (`operation`, an eshkol_operations_t tagged by its own `op` field).
 * `inferred_hott_type` caches the type checker's result (0 = not yet
 * checked); `line`/`column` give 1-based source location for diagnostics
 * (0 = unknown).
 */
typedef struct eshkol_ast {
    eshkol_type_t type;
    union {
        void *untyped_data;
        uint8_t uint8_val;
        uint16_t uint16_val;
        uint32_t uint32_val;
        uint64_t uint64_val;
        int8_t int8_val;
        int16_t int16_val;
        int32_t int32_val;
        int64_t int64_val;
        double double_val;
        struct {
            char *ptr;
            uint64_t size;
        } str_val;
        struct {
            char *id;
            uint8_t is_lambda;
            eshkol_operations_t *func_commands;
            struct eshkol_ast *variables;
            uint64_t num_variables;
            uint64_t size;
            uint8_t is_variadic;      // True if function accepts variable arguments
            char *rest_param;         // Name of rest parameter (for variadic functions)
            // HoTT type annotations (for inline annotations like (x : int))
            hott_type_expr_t **param_types;   // Array of parameter type annotations
            hott_type_expr_t *return_type;    // Return type annotation (optional)
        } eshkol_func;
        struct {
            char *id;
            struct eshkol_ast *data;
        } variable;
        struct {
            struct eshkol_ast *car;
            struct eshkol_ast *cdr;
        } cons_cell;
        struct {
            struct eshkol_ast *elements;
            uint64_t *dimensions;
            uint64_t num_dimensions;
            uint64_t total_elements;
        } tensor_val;
        eshkol_operations_t operation;
    };
    // HoTT type system: inferred type from type checker
    // Packed format: bits 0-15 = TypeId.id, bits 16-23 = universe level, bits 24-31 = flags
    // Value 0 means "not yet type-checked"
    uint32_t inferred_hott_type;

    // Source location for error reporting
    uint32_t line;      // 1-based line number (0 = unknown)
    uint32_t column;    // 1-based column number (0 = unknown)
} eshkol_ast_t;

// ===== Unified AST Literal Builders =====
// These set both the AST type fields AND inferred_hott_type for consistent type tracking.
// Packed HoTT TypeId format: bits 0-15 = id, bits 16-23 = universe, bits 24-31 = flags

/**
 * @brief In-place literal-node builders for eshkol_ast_t.
 *
 * Each `eshkol_ast_make_*` function sets both the AST node's `type` tag and
 * value union member AND its `inferred_hott_type` (to the corresponding
 * BuiltinTypes constant) in one step, so literal nodes are always
 * consistently type-tagged without a separate type-checking pass. @p node
 * must already be allocated; these functions only populate its fields.
 */
static inline void eshkol_ast_make_int64(eshkol_ast_t* node, int64_t val) {
    node->type = ESHKOL_INT64;
    node->int64_val = val;
    node->inferred_hott_type = 13; // BuiltinTypes::Int64
}

static inline void eshkol_ast_make_double(eshkol_ast_t* node, double val) {
    node->type = ESHKOL_DOUBLE;
    node->double_val = val;
    node->inferred_hott_type = 16; // BuiltinTypes::Float64
}

static inline void eshkol_ast_make_bool(eshkol_ast_t* node, bool val) {
    node->type = ESHKOL_BOOL;
    node->int64_val = val ? 1 : 0;
    node->inferred_hott_type = 25; // BuiltinTypes::Boolean
}

static inline void eshkol_ast_make_char(eshkol_ast_t* node, int64_t val) {
    node->type = ESHKOL_CHAR;
    node->int64_val = val;
    node->inferred_hott_type = 22; // BuiltinTypes::Char
}

static inline void eshkol_ast_make_null(eshkol_ast_t* node) {
    node->type = ESHKOL_NULL;
    node->int64_val = 0;
    node->inferred_hott_type = 26; // BuiltinTypes::Null
}

static inline void eshkol_ast_make_string(eshkol_ast_t* node, const char* ptr, uint64_t size) {
    node->type = ESHKOL_STRING;
    node->str_val.ptr = (char*)ptr;
    node->str_val.size = size;
    node->inferred_hott_type = 21; // BuiltinTypes::String
}

static inline void eshkol_ast_make_symbol(eshkol_ast_t* node, const char* ptr, uint64_t size) {
    node->type = ESHKOL_SYMBOL;
    node->str_val.ptr = (char*)ptr;
    node->str_val.size = size;
    node->inferred_hott_type = 27; // BuiltinTypes::Symbol
}

/**
 * @brief Recursively release resources owned by an AST node (not the node itself).
 * @param ast Node to clean up; may be modified in place.
 */
void eshkol_ast_clean(eshkol_ast_t *ast);
/**
 * @brief Print a human-readable, indented dump of an AST subtree (debugging aid).
 * @param ast Root of the subtree to print.
 * @param indent Current indentation level (spaces per nesting level), 0 for the top call.
 */
void eshkol_ast_pretty_print(const eshkol_ast_t *ast, int indent);

// Symbolic differentiation AST helpers
/**
 * @brief Allocate a zero-initialized eshkol_ast_t for use in symbolic-differentiation construction.
 * @return Newly allocated AST node.
 */
eshkol_ast_t* eshkol_alloc_symbolic_ast(void);
/**
 * @brief Build a variable-reference AST node.
 * @param name Variable name.
 * @return Newly allocated AST node referencing @p name.
 */
eshkol_ast_t* eshkol_make_var_ast(const char* name);
/**
 * @brief Build an integer literal AST node.
 * @param value Integer value.
 * @return Newly allocated AST node.
 */
eshkol_ast_t* eshkol_make_int_ast(int64_t value);
/**
 * @brief Build a double literal AST node.
 * @param value Double value.
 * @return Newly allocated AST node.
 */
eshkol_ast_t* eshkol_make_double_ast(double value);
/**
 * @brief Build a binary operator call AST node, e.g. for symbolic differentiation results.
 * @param op Operator name (e.g. "+", "*").
 * @param left Left operand subtree.
 * @param right Right operand subtree.
 * @return Newly allocated AST node representing `(op left right)`.
 */
eshkol_ast_t* eshkol_make_binary_op_ast(const char* op, eshkol_ast_t* left, eshkol_ast_t* right);
/**
 * @brief Build a unary function call AST node, e.g. for symbolic differentiation results.
 * @param func Function name (e.g. "sin", "exp").
 * @param arg Argument subtree.
 * @return Newly allocated AST node representing `(func arg)`.
 */
eshkol_ast_t* eshkol_make_unary_call_ast(const char* func, eshkol_ast_t* arg);
/**
 * @brief Deep-copy an AST subtree.
 * @param ast Subtree to copy.
 * @return Newly allocated, independent copy of @p ast.
 */
eshkol_ast_t* eshkol_copy_ast(const eshkol_ast_t* ast);

// REPL display helper
/**
 * @brief Wrap a top-level expression so the REPL displays its resulting value.
 * @param expr Expression to wrap.
 * @return A new AST node that evaluates @p expr and displays its result.
 */
eshkol_ast_t* eshkol_wrap_with_display(eshkol_ast_t* expr);

// ===== HoTT Type Expression Helpers =====
// Create primitive type expressions
/**
 * @brief Constructors for the primitive (payload-free) hott_type_expr_t kinds.
 *
 * Each returns a newly allocated hott_type_expr_t with `kind` set to the
 * corresponding HOTT_TYPE_* constant (integer, real, boolean, string,
 * char, symbol, null, the `any` top type, and the `nothing` bottom type).
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_integer_type(void);
hott_type_expr_t* hott_make_real_type(void);
hott_type_expr_t* hott_make_boolean_type(void);
hott_type_expr_t* hott_make_string_type(void);
hott_type_expr_t* hott_make_char_type(void);
hott_type_expr_t* hott_make_symbol_type(void);
hott_type_expr_t* hott_make_null_type(void);
hott_type_expr_t* hott_make_any_type(void);
hott_type_expr_t* hott_make_nothing_type(void);

// Create type variables
/**
 * @brief Construct a HOTT_TYPE_VAR type expression (a type variable, e.g. 'a in forall).
 * @param name Type variable name.
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_type_var(const char* name);

// Create compound types
/**
 * @brief Construct a HOTT_TYPE_ARROW function type `(-> param_types... return_type)`.
 * @param param_types Array of parameter type expressions.
 * @param num_params Number of entries in @p param_types.
 * @param return_type Return type expression.
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_arrow_type(hott_type_expr_t** param_types, uint64_t num_params, hott_type_expr_t* return_type);
/**
 * @brief Construct a HOTT_TYPE_LIST type expression `(list element_type)`.
 * @param element_type Element type.
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_list_type(hott_type_expr_t* element_type);
/**
 * @brief Construct a HOTT_TYPE_VECTOR type expression `(vector element_type)`.
 * @param element_type Element type.
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_vector_type(hott_type_expr_t* element_type);
/**
 * @brief Construct a HOTT_TYPE_TENSOR type expression `(tensor element_type)`.
 * @param element_type Element type.
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_tensor_type(hott_type_expr_t* element_type);
/**
 * @brief Construct a HOTT_TYPE_POINTER type expression `(ptr element_type)`.
 * @param element_type Pointee type.
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_pointer_type(hott_type_expr_t* element_type);
/**
 * @brief Construct a HOTT_TYPE_PAIR type expression `(pair left right)` (cons cell type).
 * @param left Car type.
 * @param right Cdr type.
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_pair_type(hott_type_expr_t* left, hott_type_expr_t* right);
/**
 * @brief Construct a HOTT_TYPE_PRODUCT type expression `(* left right)`.
 * @param left Left component type.
 * @param right Right component type.
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_product_type(hott_type_expr_t* left, hott_type_expr_t* right);
/**
 * @brief Construct a HOTT_TYPE_SUM type expression `(+ left right)` (either type).
 * @param left Left alternative type.
 * @param right Right alternative type.
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_sum_type(hott_type_expr_t* left, hott_type_expr_t* right);
/**
 * @brief Construct a HOTT_TYPE_FORALL type expression `(forall (type_vars...) body)`.
 * @param type_vars Array of quantified type-variable names.
 * @param num_vars Number of entries in @p type_vars.
 * @param body Body type expression under the quantifier.
 * @return Newly allocated type expression.
 */
hott_type_expr_t* hott_make_forall_type(char** type_vars, uint64_t num_vars, hott_type_expr_t* body);

// Copy and free type expressions
/**
 * @brief Deep-copy a type expression.
 * @param type Type expression to copy.
 * @return Newly allocated, independent copy.
 */
hott_type_expr_t* hott_copy_type_expr(const hott_type_expr_t* type);
/**
 * @brief Recursively free a type expression and its owned children.
 * @param type Type expression to free (may be NULL, a no-op).
 */
void hott_free_type_expr(hott_type_expr_t* type);

// Type expression to string (for display/error messages)
/**
 * @brief Render a type expression as a human-readable string, for display/error messages.
 * @param type Type expression to render.
 * @return Newly allocated, NUL-terminated string; caller owns and must free it.
 */
char* hott_type_to_string(const hott_type_expr_t* type);

// ===== Inferred HoTT Type Helpers =====
// Pack/unpack TypeId to/from uint32_t for AST storage
// Format: bits 0-15 = id, bits 16-23 = universe, bits 24-31 = flags

/**
 * @brief Pack a TypeId into the uint32_t format stored in eshkol_ast_t::inferred_hott_type.
 * @param id Type identifier (bits 0-15).
 * @param universe Universe level (bits 16-23).
 * @param flags Type flags (bits 24-31).
 * @return The packed uint32_t value.
 */
static inline uint32_t hott_pack_type_id(uint16_t id, uint8_t universe, uint8_t flags) {
    return (uint32_t)id | ((uint32_t)universe << 16) | ((uint32_t)flags << 24);
}

/**
 * @brief Extract the type identifier from a packed HoTT TypeId.
 * @param packed Value produced by hott_pack_type_id() (or eshkol_ast_t::inferred_hott_type).
 * @return The id component (bits 0-15).
 */
static inline uint16_t hott_unpack_type_id(uint32_t packed) {
    return (uint16_t)(packed & 0xFFFF);
}

/**
 * @brief Extract the universe level from a packed HoTT TypeId.
 * @param packed Value produced by hott_pack_type_id().
 * @return The universe component (bits 16-23).
 */
static inline uint8_t hott_unpack_universe(uint32_t packed) {
    return (uint8_t)((packed >> 16) & 0xFF);
}

/**
 * @brief Extract the flags byte from a packed HoTT TypeId.
 * @param packed Value produced by hott_pack_type_id().
 * @return The flags component (bits 24-31).
 */
static inline uint8_t hott_unpack_flags(uint32_t packed) {
    return (uint8_t)((packed >> 24) & 0xFF);
}

/**
 * @brief Check whether a packed HoTT TypeId represents "already type-checked".
 * @param packed Value to test (e.g. eshkol_ast_t::inferred_hott_type).
 * @return Nonzero if @p packed is non-zero (a type has been assigned), zero if unset.
 */
static inline int hott_type_is_set(uint32_t packed) {
    return packed != 0;
}

#ifdef __cplusplus
};

/**
 * @brief Parse the next top-level S-expression from an open file stream.
 * @param in_file Input file stream positioned at the next form to parse.
 * @return The parsed AST node (type ESHKOL_INVALID or similar sentinel at end of input,
 *         per the parser's end-of-stream convention).
 */
eshkol_ast_t eshkol_parse_next_ast(std::ifstream &in_file);

/**
 * @brief Parse the next top-level S-expression from any input stream.
 *
 * Generalizes eshkol_parse_next_ast() to any std::istream (e.g.
 * std::istringstream), which the stdlib loader uses to parse
 * in-memory/embedded source text.
 * @param in_stream Input stream positioned at the next form to parse.
 * @return The parsed AST node.
 */
eshkol_ast_t eshkol_parse_next_ast_from_stream(std::istream &in_stream);

// Reset cumulative file line/column counter — call before parsing a new file
// so the next eshkol_parse_next_ast call starts at file line 1, column 1.
/**
 * @brief Reset the parser's cumulative line/column counter to line 1, column 1.
 *
 * Call before parsing a new file so that source-location tracking in
 * subsequent eshkol_parse_next_ast() calls starts fresh instead of
 * continuing to accumulate from a previously parsed file/stream.
 */
extern "C" void eshkol_reset_parse_line_counter(void);

#endif


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_AST_H */
