#ifndef ESHKOL_MACRO_H
#define ESHKOL_MACRO_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "eshkol_value.h"

#ifdef __cplusplus
extern "C" {
#endif

struct eshkol_ast;
typedef struct eshkol_ast eshkol_ast_t;
struct eshkol_operation;
typedef struct eshkol_operation eshkol_operation_t;

// ═══════════════════════════════════════════════════════════════════════════

/**
 * @brief Kind discriminator for eshkol_macro_pattern_t elements.
 */
typedef enum {
    MACRO_PAT_LITERAL,      // Literal identifier that must match exactly
    MACRO_PAT_VARIABLE,     // Pattern variable to capture value
    MACRO_PAT_ELLIPSIS,     // Ellipsis marker for repetition (...)
    MACRO_PAT_LIST,         // Nested list pattern
    MACRO_PAT_IMPROPER      // Improper list pattern (x . rest)
} macro_pattern_type_t;

// Forward declaration
struct eshkol_macro_pattern;

/**
 * @brief One element of a `syntax-rules` macro pattern.
 *
 * `type` selects the active union member: `identifier` for a literal or
 * pattern-variable name, or `list` for a nested (possibly improper)
 * sub-pattern. `followed_by_ellipsis` marks elements matched repeatedly
 * via a following `...`.
 */
typedef struct eshkol_macro_pattern {
    macro_pattern_type_t type;
    union {
        // MACRO_PAT_LITERAL / MACRO_PAT_VARIABLE
        char *identifier;

        // MACRO_PAT_LIST / MACRO_PAT_IMPROPER
        struct {
            struct eshkol_macro_pattern **elements;
            uint64_t num_elements;
            struct eshkol_macro_pattern *rest;  // For improper lists only
        } list;
    };
    uint8_t followed_by_ellipsis;  // True if this element is followed by ...
} eshkol_macro_pattern_t;

/**
 * @brief Kind discriminator for eshkol_macro_template_t elements.
 */
typedef enum {
    MACRO_TPL_LITERAL,      // Literal value (copied as-is)
    MACRO_TPL_VARIABLE,     // Pattern variable reference
    MACRO_TPL_LIST,         // List constructor
    MACRO_TPL_ELLIPSIS      // Repetition of preceding element
} macro_template_type_t;

// Forward declaration
struct eshkol_macro_template;

/**
 * @brief One element of a `syntax-rules` macro expansion template.
 *
 * `type` selects the active union member: a literal AST subtree copied
 * as-is, a reference to a captured pattern variable, or a nested list
 * constructor. `followed_by_ellipsis` marks elements expanded once per
 * capture of a preceding ellipsis pattern variable.
 */
typedef struct eshkol_macro_template {
    macro_template_type_t type;
    union {
        // MACRO_TPL_LITERAL
        struct eshkol_ast *literal;

        // MACRO_TPL_VARIABLE
        char *variable_name;

        // MACRO_TPL_LIST
        struct {
            struct eshkol_macro_template **elements;
            uint64_t num_elements;
        } list;
    };
    uint8_t followed_by_ellipsis;  // True if template element followed by ...
} eshkol_macro_template_t;

/**
 * @brief One `(pattern template)` rewrite rule of a `syntax-rules` macro.
 */
typedef struct eshkol_macro_rule {
    eshkol_macro_pattern_t *pattern;    // Pattern to match (including macro name)
    eshkol_macro_template_t *template_; // Template for expansion
} eshkol_macro_rule_t;

/**
 * @brief Full macro definition: `(define-syntax name (syntax-rules (literals...) rules...))`.
 */
typedef struct eshkol_macro_def {
    char *name;                         // Macro name
    char **literals;                    // Literal identifiers that must match exactly
    uint64_t num_literals;
    eshkol_macro_rule_t *rules;         // Array of rules
    uint64_t num_rules;
} eshkol_macro_def_t;

// ===== END MACRO SYSTEM =====

/**
 * @brief Operation/special-form tag for eshkol_ast_t's `operation` union member.
 *
 * Selects which member of eshkol_operations_t's union holds the operands
 * for a given AST node: core language forms (if/define/let/lambda/...),
 * R7RS special forms (guard, call/cc, dynamic-wind, case-lambda, ...),
 * OALR memory-management forms (with-region/owned/move/borrow/shared/
 * weak-ref), automatic-differentiation operators (derivative/gradient/
 * jacobian/hessian/taylor/...), the HoTT type-annotation forms, and the
 * neuro-symbolic consciousness-engine and differentiable-memory (DNC/SDNC)
 * primitives. New members are appended at the end of relevant groups to
 * preserve on-disk/ABI ordering where noted.
 */


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_MACRO_H */
