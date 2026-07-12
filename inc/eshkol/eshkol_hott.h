#ifndef ESHKOL_HOTT_H
#define ESHKOL_HOTT_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "eshkol_value.h"

#ifdef __cplusplus
extern "C" {
#endif

// Homotopy Type Theory inspired type expressions for static type checking

/**
 * @brief Kind discriminator for hott_type_expr_t.
 *
 * Selects which member of hott_type_expr_t's union is active: none for the
 * primitive types, `var_name` for HOTT_TYPE_VAR, `arrow`/`forall`/
 * `container`/`pair`/`sum`/`universe` for the corresponding compound kinds.
 * HOTT_TYPE_DEPENDENT/_PATH are reserved for future dependent-type support.
 */
typedef enum {
    HOTT_TYPE_INVALID,
    // Primitive types
    HOTT_TYPE_INTEGER,      // integer
    HOTT_TYPE_REAL,         // real (double)
    HOTT_TYPE_NUMBER,       // number (supertype of integer and real)
    HOTT_TYPE_BOOLEAN,      // boolean
    HOTT_TYPE_STRING,       // string
    HOTT_TYPE_CHAR,         // char
    HOTT_TYPE_SYMBOL,       // symbol
    HOTT_TYPE_NULL,         // null (empty list)
    HOTT_TYPE_ANY,          // any (top type)
    HOTT_TYPE_NOTHING,      // nothing (bottom type)
    // Compound types
    HOTT_TYPE_ARROW,        // (-> a b) function type
    HOTT_TYPE_PRODUCT,      // (* a b) product type
    HOTT_TYPE_SUM,          // (+ a b) sum type (either)
    HOTT_TYPE_LIST,         // (list a) list type
    HOTT_TYPE_VECTOR,       // (vector a) vector type
    HOTT_TYPE_TENSOR,       // (tensor a) tensor type (multi-dimensional array)
    HOTT_TYPE_POINTER,      // (ptr a) raw pointer type
    HOTT_TYPE_PAIR,         // (pair a b) cons pair type
    // Polymorphic types
    HOTT_TYPE_VAR,          // type variable (e.g., 'a in forall)
    HOTT_TYPE_FORALL,       // (forall (a b ...) type) universal quantification
    // Advanced types (future)
    HOTT_TYPE_DEPENDENT,    // dependent type (Pi type)
    HOTT_TYPE_PATH,         // path type (identity type)
    HOTT_TYPE_UNIVERSE      // universe level
} hott_type_kind_t;

// Forward declaration
struct hott_type_expr;

/**
 * @brief Homotopy-Type-Theory-inspired static type expression.
 *
 * Represents the type of an Eshkol expression for the optional static type
 * checker: `kind` selects the active union member. Primitive kinds (e.g.
 * HOTT_TYPE_INTEGER) carry no payload; compound kinds recursively reference
 * child hott_type_expr_t nodes (arrow/forall/container/pair/sum), and
 * HOTT_TYPE_VAR carries a type-variable name. See the hott_make_*()
 * constructor functions below for the supported ways to build one.
 */
typedef struct hott_type_expr {
    hott_type_kind_t kind;
    union {
        // For type variables: the variable name
        char* var_name;

        // For arrow types: domain -> codomain
        struct {
            struct hott_type_expr** param_types;  // Array of parameter types
            uint64_t num_params;
            struct hott_type_expr* return_type;
        } arrow;

        // For forall types: quantified variables and body
        struct {
            char** type_vars;           // Array of type variable names
            uint64_t num_vars;
            struct hott_type_expr* body;
        } forall;

        // For compound single-parameter types: list, vector, tensor, ptr
        struct {
            struct hott_type_expr* element_type;
        } container;

        // For pair/product types
        struct {
            struct hott_type_expr* left;
            struct hott_type_expr* right;
        } pair;

        // For sum types
        struct {
            struct hott_type_expr* left;
            struct hott_type_expr* right;
        } sum;

        // For universe types
        struct {
            uint32_t level;  // Universe level (U0, U1, U2, ...)
        } universe;
    };
} hott_type_expr_t;

// Helper macros for type construction
/**
 * @brief Construct a primitive hott_type_expr_t value with no payload.
 * @param kind One of the primitive hott_type_kind_t values (e.g. HOTT_TYPE_INTEGER).
 * @return A compound literal hott_type_expr_t with `kind` set and the union left zero-initialized.
 */
#define HOTT_MAKE_PRIMITIVE(kind) ((hott_type_expr_t){.kind = (kind)})

// ===== END HoTT TYPE SYSTEM =====

// ===== PATTERN MATCHING SYSTEM =====

/**
 * @brief Kind discriminator for eshkol_pattern_t, one per `match` pattern form.
 */
typedef enum {
    PATTERN_INVALID,
    PATTERN_LITERAL,      // Literal value: 42, "hello", #t
    PATTERN_VARIABLE,     // Variable binding: x, y, z
    PATTERN_WILDCARD,     // Wildcard: _
    PATTERN_CONS,         // Cons pattern: (cons car cdr)
    PATTERN_LIST,         // List pattern: (list x y z ...)
    PATTERN_PREDICATE,    // Predicate: (? pred)
    PATTERN_OR            // Or pattern: (or p1 p2)
} pattern_type_t;

// Forward declarations
struct eshkol_pattern;
struct eshkol_ast;

/**
 * @brief One pattern node in a `match` expression's pattern tree.
 *
 * `type` selects the active union member, mirroring pattern_type_t's forms:
 * literals, variable bindings, wildcards, cons/list destructuring,
 * predicate guards (with optional binding), and alternation (`or`).
 */
typedef struct eshkol_pattern {
    pattern_type_t type;
    union {
        // PATTERN_LITERAL: stores the literal value
        struct {
            struct eshkol_ast *value;
        } literal;

        // PATTERN_VARIABLE: variable name to bind
        struct {
            char *name;
        } variable;

        // PATTERN_CONS: (cons car_pat cdr_pat)
        struct {
            struct eshkol_pattern *car_pattern;
            struct eshkol_pattern *cdr_pattern;
        } cons;

        // PATTERN_LIST: (list pat1 pat2 ...)
        struct {
            struct eshkol_pattern **patterns;
            uint64_t num_patterns;
        } list;

        // PATTERN_PREDICATE: (? pred-expr) or (? pred-expr name)
        // If `name` is non-null, the matched value is bound to that
        // variable in the clause body (Racket-style guard-with-binding).
        struct {
            struct eshkol_ast *predicate;
            char *binding_name;  // optional; null = no binding
        } predicate;

        // PATTERN_OR: (or pat1 pat2 ...)
        struct {
            struct eshkol_pattern **patterns;
            uint64_t num_patterns;
        } or_pat;
    };
} eshkol_pattern_t;

/**
 * @brief One `(pattern [when guard] body)` clause of a `match` expression.
 */
typedef struct eshkol_match_clause {
    eshkol_pattern_t *pattern;      // Pattern to match
    struct eshkol_ast *guard;       // Optional (when ...) guard expression
    struct eshkol_ast *body;        // Body expression
} eshkol_match_clause_t;

// ===== END PATTERN MATCHING SYSTEM =====


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_HOTT_H */
