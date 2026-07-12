#ifndef ESHKOL_TYPE_H
#define ESHKOL_TYPE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief AST/parser-level value type tag used by eshkol_ast_t.
 *
 * Distinguishes the primitive and compound value kinds the frontend AST can
 * hold (as opposed to eshkol_value_type_t, which tags runtime tagged
 * values). ESHKOL_BIGNUM_LITERAL and ESHKOL_SYMBOL literals are stored as
 * strings until lowered by codegen.
 */
typedef enum {
    ESHKOL_INVALID,
    ESHKOL_UNTYPED,
    ESHKOL_UINT8,
    ESHKOL_UINT16,
    ESHKOL_UINT32,
    ESHKOL_UINT64,
    ESHKOL_INT8,
    ESHKOL_INT16,
    ESHKOL_INT32,
    ESHKOL_INT64,
    ESHKOL_DOUBLE,
    ESHKOL_STRING,
    ESHKOL_FUNC,
    ESHKOL_VAR,
    ESHKOL_OP,
    ESHKOL_CONS,
    ESHKOL_NULL,
    ESHKOL_TENSOR,
    ESHKOL_CHAR,
    ESHKOL_BOOL,
    ESHKOL_BIGNUM_LITERAL,  // Integer literal too large for int64 (stored as string)
    ESHKOL_SYMBOL           // Symbol literal (stored as string)
} eshkol_type_t;


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TYPE_H */
