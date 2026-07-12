#ifndef ESHKOL_VALUE_H
#define ESHKOL_VALUE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "eshkol_version.h"
#include "eshkol_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════
// VALUE TYPE TAGS - 8-bit type field with subtype headers for pointers
// ═══════════════════════════════════════════════════════════════════════════
/**
 * @brief Runtime tag stored in eshkol_tagged_value_t::type.
 *
 * Identifies the kind of value carried by a tagged value. Values 0-10 are
 * immediates or consolidated pointer kinds that resolve to a concrete
 * representation via the object header's subtype (see heap_subtype_t /
 * callable_subtype_t); values 16-19 are multimedia resource kinds; values
 * 32+ are deprecated single-purpose pointer tags kept only so the display
 * system can render data produced before the M1 pointer-consolidation
 * migration. New code must not use the deprecated values.
 */
typedef enum {
    // ═══════════════════════════════════════════════════════════════════════
    // IMMEDIATE VALUES (0-7) - data stored directly in tagged value
    // No heap allocation, no header needed
    // ═══════════════════════════════════════════════════════════════════════
    ESHKOL_VALUE_NULL        = 0,   // Empty/null value
    ESHKOL_VALUE_INT64       = 1,   // 64-bit signed integer
    ESHKOL_VALUE_DOUBLE      = 2,   // Double-precision float
    ESHKOL_VALUE_BOOL        = 3,   // Boolean (#t/#f)
    ESHKOL_VALUE_CHAR        = 4,   // Unicode character
    ESHKOL_VALUE_SYMBOL      = 5,   // Interned symbol
    ESHKOL_VALUE_DUAL_NUMBER = 6,   // Forward-mode AD dual number
    ESHKOL_VALUE_COMPLEX     = 7,   // Complex number (real + imaginary)

    // ═══════════════════════════════════════════════════════════════════════
    // CONSOLIDATED POINTER TYPES (8-9) - subtype in object header
    // All heap-allocated objects have eshkol_object_header_t prefix
    // ═══════════════════════════════════════════════════════════════════════
    ESHKOL_VALUE_HEAP_PTR    = 8,   // Heap data: cons, string, vector, tensor, hash, exception
    ESHKOL_VALUE_CALLABLE    = 9,   // Callables: closure, lambda-sexpr, ad-node

    // Neuro-symbolic consciousness engine types
    ESHKOL_VALUE_LOGIC_VAR   = 10,  // Logic variable ?x (data = var_id : int64)

    // ═══════════════════════════════════════════════════════════════════════
    // MULTIMEDIA TYPES (16-19) - linear resources with lifecycle management
    // ═══════════════════════════════════════════════════════════════════════
    ESHKOL_VALUE_HANDLE      = 16,  // Managed resource handles
    ESHKOL_VALUE_BUFFER      = 17,  // Typed data buffers
    ESHKOL_VALUE_STREAM      = 18,  // Lazy data streams
    ESHKOL_VALUE_EVENT       = 19,  // Input/system events

    // ═══════════════════════════════════════════════════════════════════════
    // LEGACY TYPES (DEPRECATED) - For display system backward compatibility only
    // M1 Migration is COMPLETE. New code MUST use consolidated types:
    //   - HEAP_PTR (8) + subtype for: cons, string, vector, tensor, hash, exception
    //   - CALLABLE (9) + subtype for: closure, lambda-sexpr, ad-node
    // These constants are retained only for the display system to render old data.
    // DO NOT use these in new code - they will be removed in a future version.
    // ═══════════════════════════════════════════════════════════════════════
    ESHKOL_VALUE_CONS_PTR    = 32,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_CONS
    ESHKOL_VALUE_STRING_PTR  = 33,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_STRING
    ESHKOL_VALUE_VECTOR_PTR  = 34,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_VECTOR
    ESHKOL_VALUE_TENSOR_PTR  = 35,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_TENSOR
    ESHKOL_VALUE_HASH_PTR    = 36,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_HASH
    ESHKOL_VALUE_EXCEPTION   = 37,  // DEPRECATED: use HEAP_PTR + HEAP_SUBTYPE_EXCEPTION
    ESHKOL_VALUE_CLOSURE_PTR = 38,  // DEPRECATED: use CALLABLE + CALLABLE_SUBTYPE_CLOSURE
    ESHKOL_VALUE_LAMBDA_SEXPR = 39, // DEPRECATED: use CALLABLE + CALLABLE_SUBTYPE_LAMBDA_SEXPR
    ESHKOL_VALUE_AD_NODE_PTR = 40,  // DEPRECATED: use CALLABLE + CALLABLE_SUBTYPE_AD_NODE
} eshkol_value_type_t;

// Type flags for Scheme exactness tracking
#define ESHKOL_VALUE_EXACT_FLAG   0x10
#define ESHKOL_VALUE_INEXACT_FLAG 0x20

// Port type flags (OR'd into type byte with ESHKOL_VALUE_HEAP_PTR)
#define ESHKOL_PORT_INPUT_FLAG    0x10   // Input port
#define ESHKOL_PORT_OUTPUT_FLAG   0x40   // Output port
#define ESHKOL_PORT_BINARY_FLAG   0x04   // Binary port (vs textual)
#define ESHKOL_PORT_ANY_FLAG      0x50   // Mask: input (0x10) | output (0x40)

// Combined type constants for common cases
#define ESHKOL_VALUE_EXACT_INT64     (ESHKOL_VALUE_INT64 | ESHKOL_VALUE_EXACT_FLAG)
#define ESHKOL_VALUE_INEXACT_DOUBLE  (ESHKOL_VALUE_DOUBLE | ESHKOL_VALUE_INEXACT_FLAG)

/**
 * @brief Raw payload union backing eshkol_tagged_value_t and cons cell slots.
 *
 * Exactly one member is active at a time, selected by the owning tagged
 * value's `type` field. All members occupy the same 8 bytes; `raw_val`
 * exists purely to allow bit-level manipulation/zero-initialization without
 * caring which typed member is logically active.
 */
typedef union eshkol_tagged_data {
    int64_t int_val;     // Integer value
    double double_val;   // Double-precision floating point value
    uint64_t ptr_val;    // Pointer value (for cons cell pointers)
    uint64_t raw_val;    // Raw 64-bit value for manipulation
} eshkol_tagged_data_t;

/**
 * @brief Universal runtime representation of an Eshkol value.
 *
 * Every Scheme value flowing through the compiled runtime and the embedding
 * API is carried in this 16-byte struct: `type` selects the active member of
 * `data` (see eshkol_value_type_t), `flags` records exactness and other
 * per-value bits (ESHKOL_VALUE_EXACT_FLAG / ESHKOL_VALUE_INEXACT_FLAG /
 * port flags), and `reserved` is unused padding kept available for future
 * tag bits. For heap-allocated kinds (HEAP_PTR, CALLABLE, ...) `data.ptr_val`
 * points past an eshkol_object_header_t that carries the concrete subtype.
 */
typedef struct eshkol_tagged_value {
    uint8_t type;        // Value type (eshkol_value_type_t)
    uint8_t flags;       // Exactness and other flags
    uint16_t reserved;   // Reserved for future use
    union {
        int64_t int_val;
        double double_val;
        uint64_t ptr_val;
        uint64_t raw_val;   // For raw manipulation and zero-initialization
    } data;
} eshkol_tagged_value_t;

// Compile-time size validation for tagged values
ESHKOL_STATIC_ASSERT(sizeof(eshkol_tagged_value_t) <= 16,
                     "Tagged value must fit in 16 bytes for efficiency");

/**
 * @brief Dual number for forward-mode automatic differentiation.
 *
 * Carries a function value and its derivative together so that arithmetic
 * on dual numbers propagates derivatives via the chain rule without a
 * separate backward pass. Used as the scalar unit of forward-mode AD
 * (see also esh_taylor_t for the arbitrary-order generalization).
 */
typedef struct eshkol_dual_number {
    double value;       // f(x) - the function value
    double derivative;  // f'(x) - the derivative value
} eshkol_dual_number_t;

// Compile-time size validation for dual numbers
ESHKOL_STATIC_ASSERT(sizeof(eshkol_dual_number_t) == 16,
                     "Dual number must be 16 bytes for cache efficiency");

// ───────────────────────────────────────────────────────────────────────────
// TAYLOR TOWER  (arbitrary-order forward-mode AD — ESH-0186, docs/design/AD_TAYLOR_TOWER.md)
// ───────────────────────────────────────────────────────────────────────────
// A univariate truncated-Taylor series carried as its coefficient array:
//     f(x0 + t) = sum_{k=0..K} c[k] * t^k        =>   f^(n)(x0) = n! * c[n].
// Stored behind a tagged HEAP_PTR with subtype HEAP_SUBTYPE_TAYLOR. Mirrors
// HEAP_SUBTYPE_TENSOR's homogeneous-double storage: the header is followed by
// (order_k + 1) doubles. `flags` reserves a coefficient-type field and a
// perturbation-confusion EPOCH tag so nested `derivative` levels never mix
// (§5a). Differentiating the variable seeds {x0, 1, 0, ...}.
typedef struct esh_taylor {
    uint32_t order_k;   // highest coefficient index K (series has K+1 entries)
    uint32_t flags;     // packed: COEFF_MASK[0..7] | RESERVED0[8..15] | EPOCH_TAG[16..31]
    double   c[];       // coefficient storage c[0..order_k] (COEFF_F64)
} esh_taylor_t;

// esh_taylor_t.flags bitfield accessors (§4 of the design).
#define ESH_TAYLOR_COEFF_MASK    0x000000FFu  // coefficient type: 0=F64, 1=RATIONAL (P7 adds TENSOR)
#define ESH_TAYLOR_COEFF_F64     0u
// P6 (ESH-0191): exact-coefficient towers. When COEFF_RATIONAL is set, the
// `c[]` storage declared as `double c[]` above is REINTERPRETED as
// `eshkol_tagged_value_t c[order_k+1]` (each entry an exact int64/bignum/
// rational tagged value, produced by Eshkol's existing exact numeric tower)
// instead of raw doubles. This is safe because `c`'s offset (right after
// order_k/flags, 8 bytes in) is already 8-byte aligned, matching
// alignof(eshkol_tagged_value_t); accessors in lib/core/runtime_taylor.c
// never raw-index across coefficient types (design section 4/12).
#define ESH_TAYLOR_COEFF_RATIONAL 1u
// P5 (ESH-0190) reverse-over-Taylor: a tower may carry a parallel first-order
// "seed tangent" series alongside its value series. When ESH_TAYLOR_TANGENT_FLAG
// is set the coefficient storage holds 2*(K+1) doubles: c[0..K] values followed
// by c[K+1..2K+1] = d(value_k)/d(reverse-seed). This is the tower analogue of
// the 8-jet's ep-derivative half (docs/design/AD_TAYLOR_TOWER.md §8). Lives in
// the RESERVED0 byte (bit 8); orthogonal to COEFF_MASK and EPOCH_TAG.
#define ESH_TAYLOR_TANGENT_FLAG  0x00000100u
#define ESH_TAYLOR_HAS_TANGENT(fl) (((fl) & ESH_TAYLOR_TANGENT_FLAG) != 0u)
#define ESH_TAYLOR_EPOCH_SHIFT   16u
#define ESH_TAYLOR_EPOCH_MASK    0xFFFF0000u  // perturbation-confusion tag (bits 16..31)
#define ESH_TAYLOR_GET_EPOCH(fl) (((fl) & ESH_TAYLOR_EPOCH_MASK) >> ESH_TAYLOR_EPOCH_SHIFT)
#define ESH_TAYLOR_MK_FLAGS(coeff, epoch) \
    (((uint32_t)(coeff) & ESH_TAYLOR_COEFF_MASK) | \
     (((uint32_t)(epoch) << ESH_TAYLOR_EPOCH_SHIFT) & ESH_TAYLOR_EPOCH_MASK))

/**
 * @brief Complex number for signal processing, FFT, and complex analysis.
 *
 * Stores the real and imaginary components as IEEE 754 double-precision
 * floats. Complex-valued tagged values (ESHKOL_VALUE_COMPLEX) store a
 * pointer to a heap-allocated instance of this struct.
 */
typedef struct eshkol_complex_number {
    double real;        // Real component (ℜ)
    double imag;        // Imaginary component (𝕴)
} eshkol_complex_number_t;

// Compile-time size validation for complex numbers
ESHKOL_STATIC_ASSERT(sizeof(eshkol_complex_number_t) == 16,
                     "Complex number must be 16 bytes for cache efficiency");

// Helper functions for tagged value manipulation
/**
 * @brief Construct a tagged value wrapping a 64-bit signed integer.
 * @param val Integer payload.
 * @param exact Whether the value is R7RS-exact; sets ESHKOL_VALUE_EXACT_FLAG.
 * @return A tagged value of type ESHKOL_VALUE_INT64.
 */
static inline eshkol_tagged_value_t eshkol_make_int64(int64_t val, bool exact) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_INT64;
    result.flags = exact ? ESHKOL_VALUE_EXACT_FLAG : 0;
    result.reserved = 0;
    result.data.int_val = val;
    return result;
}

/**
 * @brief Construct a tagged value wrapping an inexact double.
 * @param val Double-precision payload.
 * @return A tagged value of type ESHKOL_VALUE_DOUBLE with the inexact flag set.
 */
static inline eshkol_tagged_value_t eshkol_make_double(double val) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_DOUBLE;
    result.flags = ESHKOL_VALUE_INEXACT_FLAG;
    result.reserved = 0;
    result.data.double_val = val;
    return result;
}

/**
 * @brief Construct a tagged value wrapping a raw pointer with an explicit type tag.
 * @param ptr Pointer value, stored in the tagged value's ptr_val union member.
 * @param type Type tag to assign (e.g. ESHKOL_VALUE_HEAP_PTR, ESHKOL_VALUE_CALLABLE).
 * @return A tagged value with `flags`/`reserved` cleared and `data.ptr_val = ptr`.
 */
static inline eshkol_tagged_value_t eshkol_make_ptr(uint64_t ptr, uint8_t type) {
    eshkol_tagged_value_t result;
    result.type = type;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = ptr;
    return result;
}

/**
 * @brief Construct a tagged value wrapping a heap-allocated complex number.
 * @param ptr Pointer to an eshkol_complex_number_t.
 * @return A tagged value of type ESHKOL_VALUE_COMPLEX (always inexact).
 */
static inline eshkol_tagged_value_t eshkol_make_complex(uint64_t ptr) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_COMPLEX;
    result.flags = ESHKOL_VALUE_INEXACT_FLAG;  // Complex numbers are always inexact
    result.reserved = 0;
    result.data.ptr_val = ptr;
    return result;
}

/**
 * @brief Read the integer payload from a tagged value.
 * @param val Tagged value; caller must ensure its type stores int64 data.
 * @return The raw `data.int_val` field, without any type checking.
 */
static inline int64_t eshkol_unpack_int64(const eshkol_tagged_value_t* val) {
    return val->data.int_val;
}

/**
 * @brief Read the double payload from a tagged value.
 * @param val Tagged value; caller must ensure its type stores double data.
 * @return The raw `data.double_val` field, without any type checking.
 */
static inline double eshkol_unpack_double(const eshkol_tagged_value_t* val) {
    return val->data.double_val;
}

/**
 * @brief Read the pointer payload from a tagged value.
 * @param val Tagged value; caller must ensure its type stores pointer data.
 * @return The raw `data.ptr_val` field, without any type checking.
 */
static inline uint64_t eshkol_unpack_ptr(const eshkol_tagged_value_t* val) {
    return val->data.ptr_val;
}


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_VALUE_H */
