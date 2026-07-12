#ifndef ESHKOL_TYPECHECK_H
#define ESHKOL_TYPECHECK_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "eshkol_value.h"

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CHECKING MACROS - Handle both new consolidated and legacy types
// ═══════════════════════════════════════════════════════════════════════════

// Immediate type checks (no masking needed for new types)
#define ESHKOL_IS_NULL_TYPE(type)        ((type) == ESHKOL_VALUE_NULL)
#define ESHKOL_IS_INT64_TYPE(type)       ((type) == ESHKOL_VALUE_INT64)
#define ESHKOL_IS_DOUBLE_TYPE(type)      ((type) == ESHKOL_VALUE_DOUBLE)
#define ESHKOL_IS_BOOL_TYPE(type)        ((type) == ESHKOL_VALUE_BOOL)
#define ESHKOL_IS_CHAR_TYPE(type)        ((type) == ESHKOL_VALUE_CHAR)
#define ESHKOL_IS_SYMBOL_TYPE(type)      ((type) == ESHKOL_VALUE_SYMBOL)
#define ESHKOL_IS_DUAL_NUMBER_TYPE(type) ((type) == ESHKOL_VALUE_DUAL_NUMBER)
#define ESHKOL_IS_COMPLEX_TYPE(type)     ((type) == ESHKOL_VALUE_COMPLEX)

// Storage type checks - for cons cell setters that take int64 or double storage
// INT64 storage: INT64, BOOL, CHAR, SYMBOL (types that use int_val in the union)
// Also includes legacy pointer types (32+) and consolidated types (HEAP_PTR, CALLABLE)
// which store pointer addresses as int64
#define ESHKOL_IS_INT_STORAGE_TYPE(type) ((type) == ESHKOL_VALUE_INT64 || \
                                          (type) == ESHKOL_VALUE_BOOL || \
                                          (type) == ESHKOL_VALUE_CHAR || \
                                          (type) == ESHKOL_VALUE_SYMBOL || \
                                          (type) == ESHKOL_VALUE_HEAP_PTR || \
                                          (type) == ESHKOL_VALUE_CALLABLE || \
                                          (type) >= 32)

// Consolidated type checks
#define ESHKOL_IS_HEAP_PTR_TYPE(type)    ((type) == ESHKOL_VALUE_HEAP_PTR)
#define ESHKOL_IS_CALLABLE_TYPE(type)    ((type) == ESHKOL_VALUE_CALLABLE)
#define ESHKOL_IS_LOGIC_VAR_TYPE(type)   ((type) == ESHKOL_VALUE_LOGIC_VAR)

// Neuro-symbolic consciousness engine macros
#define ESHKOL_IS_LOGIC_VAR(tv)  ((tv).type == ESHKOL_VALUE_LOGIC_VAR)
#define ESHKOL_LOGIC_VAR_ID(tv)  ((tv).data.int_val)

// ───────────────────────────────────────────────────────────────────────────
// DEPRECATED Legacy type checks - for display system backward compatibility
// New code should check HEAP_PTR/CALLABLE type and read subtype from header
// ───────────────────────────────────────────────────────────────────────────
#define ESHKOL_IS_CONS_PTR_TYPE(type)    ((type) == ESHKOL_VALUE_CONS_PTR)     // DEPRECATED
#define ESHKOL_IS_STRING_PTR_TYPE(type)  ((type) == ESHKOL_VALUE_STRING_PTR)   // DEPRECATED
#define ESHKOL_IS_VECTOR_PTR_TYPE(type)  ((type) == ESHKOL_VALUE_VECTOR_PTR)   // DEPRECATED
#define ESHKOL_IS_TENSOR_PTR_TYPE(type)  ((type) == ESHKOL_VALUE_TENSOR_PTR)   // DEPRECATED
#define ESHKOL_IS_HASH_PTR_TYPE(type)    ((type) == ESHKOL_VALUE_HASH_PTR)     // DEPRECATED
#define ESHKOL_IS_EXCEPTION_TYPE(type)   ((type) == ESHKOL_VALUE_EXCEPTION)    // DEPRECATED
#define ESHKOL_IS_CLOSURE_PTR_TYPE(type) ((type) == ESHKOL_VALUE_CLOSURE_PTR)  // DEPRECATED
#define ESHKOL_IS_LAMBDA_SEXPR_TYPE(type) ((type) == ESHKOL_VALUE_LAMBDA_SEXPR) // DEPRECATED
#define ESHKOL_IS_AD_NODE_PTR_TYPE(type) ((type) == ESHKOL_VALUE_AD_NODE_PTR)  // DEPRECATED

// Combined type checks (consolidated + legacy for display compatibility)
#define ESHKOL_IS_ANY_HEAP_TYPE(type)    (ESHKOL_IS_HEAP_PTR_TYPE(type) || \
                                          ESHKOL_IS_CONS_PTR_TYPE(type) || \
                                          ESHKOL_IS_STRING_PTR_TYPE(type) || \
                                          ESHKOL_IS_VECTOR_PTR_TYPE(type) || \
                                          ESHKOL_IS_TENSOR_PTR_TYPE(type) || \
                                          ESHKOL_IS_HASH_PTR_TYPE(type) || \
                                          ESHKOL_IS_EXCEPTION_TYPE(type))

#define ESHKOL_IS_ANY_CALLABLE_TYPE(type) (ESHKOL_IS_CALLABLE_TYPE(type) || \
                                           ESHKOL_IS_CLOSURE_PTR_TYPE(type) || \
                                           ESHKOL_IS_LAMBDA_SEXPR_TYPE(type) || \
                                           ESHKOL_IS_AD_NODE_PTR_TYPE(type))

// General pointer type check: any type that stores a pointer value
#define ESHKOL_IS_ANY_PTR_TYPE(type)     (ESHKOL_IS_ANY_HEAP_TYPE(type) || \
                                          ESHKOL_IS_ANY_CALLABLE_TYPE(type))

// Exactness checking macros
#define ESHKOL_IS_EXACT(type)         (((type) & ESHKOL_VALUE_EXACT_FLAG) != 0)
#define ESHKOL_IS_INEXACT(type)       (((type) & ESHKOL_VALUE_INEXACT_FLAG) != 0)

// Type manipulation macros
#define ESHKOL_MAKE_EXACT(type)       ((type) | ESHKOL_VALUE_EXACT_FLAG)
#define ESHKOL_MAKE_INEXACT(type)     ((type) | ESHKOL_VALUE_INEXACT_FLAG)
// For new types, no masking needed. For types with exactness flags, mask them off.
#define ESHKOL_GET_BASE_TYPE(type)    ((type) & 0x3F)  // 6 bits for base type (0-63)


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TYPECHECK_H */
