#ifndef ESHKOL_CLOSURE_H
#define ESHKOL_CLOSURE_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "eshkol_object.h"

#ifdef __cplusplus
extern "C" {
#endif

// Support for lexical closures - capturing parent scope variables

// Closure environment structure (arena-allocated)
// Holds captured variables from parent scope for nested functions
//
// VARIADIC ENCODING: The num_captures field encodes both capture count and variadic info:
//   - Bits 0-15:  num_captures (up to 65535 captures)
//   - Bits 16-31: fixed_param_count (up to 65535 fixed params)
//   - Bit 63:     is_variadic flag (1 = variadic, 0 = not variadic)
//
// Use the macros below to extract/encode these values:
/**
 * @brief Accessors/constructor for the packed `num_captures` field of
 * eshkol_closure_env_t.
 *
 * The single size_t packs three logically distinct values so the header
 * stays minimal: capture count, fixed parameter count, and a variadic flag
 * (see field layout above). CLOSURE_ENV_PACK builds the packed value;
 * the GET_* macros extract each component.
 */
#define CLOSURE_ENV_GET_NUM_CAPTURES(packed) ((packed) & 0xFFFF)
#define CLOSURE_ENV_GET_FIXED_PARAMS(packed) (((packed) >> 16) & 0xFFFF)
#define CLOSURE_ENV_IS_VARIADIC(packed) (((packed) >> 63) & 1)
#define CLOSURE_ENV_PACK(num_caps, fixed_params, is_var) \
    (((size_t)(num_caps) & 0xFFFF) | \
     (((size_t)(fixed_params) & 0xFFFF) << 16) | \
     ((size_t)(is_var) << 63))

/**
 * @brief Arena-allocated environment holding a closure's captured variables.
 *
 * Laid out as a minimal packed header (`num_captures`, see the
 * CLOSURE_ENV_* macros) followed by a flexible array of the captured
 * tagged values themselves, so the whole environment is a single
 * contiguous allocation.
 */
typedef struct eshkol_closure_env {
    size_t num_captures;                  // Packed: num_captures | (fixed_params << 16) | (is_variadic << 63)
    eshkol_tagged_value_t captures[];     // Flexible array of captured values
} eshkol_closure_env_t;

// Compile-time size validation
ESHKOL_STATIC_ASSERT(sizeof(eshkol_closure_env_t) == sizeof(size_t),
                     "Closure environment header must be minimal");

// Closure return type constants (matches eshkol_value_type_t but with additional info)
// These are used to determine function behavior without calling it
#define CLOSURE_RETURN_UNKNOWN     0x00  // Return type not known (untyped lambda)
#define CLOSURE_RETURN_SCALAR      0x01  // Returns a scalar (int64 or double)
#define CLOSURE_RETURN_VECTOR      0x02  // Returns a vector/tensor
#define CLOSURE_RETURN_LIST        0x03  // Returns a list (cons cells)
#define CLOSURE_RETURN_FUNCTION    0x04  // Returns another function (higher-order)
#define CLOSURE_RETURN_BOOL        0x05  // Returns a boolean
#define CLOSURE_RETURN_STRING      0x06  // Returns a string
#define CLOSURE_RETURN_VOID        0x07  // Returns null/void

// Closure flags (stored in closure->flags field)
#define CLOSURE_FLAG_VARIADIC              0x01  // Closure accepts variadic arguments
#define CLOSURE_FLAG_NAMED                 0x02  // Closure has a bound name
#define ESHKOL_CLOSURE_FLAG_VARIADIC       CLOSURE_FLAG_VARIADIC  // Alias for consistency
#define ESHKOL_CLOSURE_FLAG_NAMED          CLOSURE_FLAG_NAMED     // Alias for consistency

/**
 * @brief Full closure object: function pointer plus captured environment.
 *
 * Allocated whenever a closure-returning form (lambda, named let, etc.)
 * produces a first-class function value. `env` may be NULL for closures
 * with no captures (top-level defines). `sexpr_ptr` links back to the
 * closure's S-expression form to support homoiconicity (e.g. `(display
 * some-lambda)`); `return_type` records enough static information
 * (CLOSURE_RETURN_*) to make dispatch decisions (e.g. for autodiff) without
 * invoking the closure.
 */
typedef struct eshkol_closure {
    uint64_t func_ptr;                    // Pointer to the lambda function
    eshkol_closure_env_t* env;            // Pointer to captured environment (may be NULL for no captures)
    uint64_t sexpr_ptr;                   // Pointer to S-expression representation for homoiconicity
    const char* name;                     // Bound name from (define name ...) or NULL for anonymous lambdas
    uint8_t return_type;                  // Return type category (CLOSURE_RETURN_*)
    uint8_t input_arity;                  // Number of expected input arguments (0-255)
    uint8_t flags;                        // Additional flags (CLOSURE_FLAG_VARIADIC, etc.)
    uint8_t reserved;                     // Padding for alignment
    uint32_t hott_type_id;                // HoTT TypeId for the return type (0 = unknown)
} eshkol_closure_t;

// Compile-time size validation for closure structure
ESHKOL_STATIC_ASSERT(sizeof(eshkol_closure_t) == 40,
                     "Closure structure must be 40 bytes for alignment");

// Helper macros for closure type queries
#define CLOSURE_RETURNS_VECTOR(closure) ((closure)->return_type == CLOSURE_RETURN_VECTOR)
#define CLOSURE_RETURNS_SCALAR(closure) ((closure)->return_type == CLOSURE_RETURN_SCALAR)
#define CLOSURE_RETURNS_FUNCTION(closure) ((closure)->return_type == CLOSURE_RETURN_FUNCTION)
#define CLOSURE_TYPE_KNOWN(closure) ((closure)->return_type != CLOSURE_RETURN_UNKNOWN)

/**
 * @brief Get a closure's static return-type category from a tagged value.
 *
 * Supports both the consolidated CALLABLE type (verifying the object
 * header's subtype is CALLABLE_SUBTYPE_CLOSURE before dereferencing) and
 * the deprecated legacy CLOSURE_PTR tag.
 *
 * @param tagged Tagged value expected to hold a closure.
 * @return One of the CLOSURE_RETURN_* constants, or CLOSURE_RETURN_UNKNOWN
 *         if @p tagged is not a closure, is NULL, or has an unrecognized
 *         subtype.
 */
static inline uint8_t eshkol_closure_get_return_type(eshkol_tagged_value_t tagged) {
    uint8_t base_type = tagged.type & 0x3F;  // Mask off flags

    // Check for consolidated CALLABLE type
    if (base_type == ESHKOL_VALUE_CALLABLE) {
        uint64_t ptr = tagged.data.ptr_val;
        if (!ptr) return CLOSURE_RETURN_UNKNOWN;
        // For CALLABLE, verify subtype is closure before reading
        eshkol_object_header_t* header = ESHKOL_GET_HEADER((void*)ptr);
        if (header->subtype != CALLABLE_SUBTYPE_CLOSURE) {
            return CLOSURE_RETURN_UNKNOWN;
        }
        eshkol_closure_t* closure = (eshkol_closure_t*)ptr;
        return closure->return_type;
    }

    // Legacy support: CLOSURE_PTR (deprecated)
    if (base_type == ESHKOL_VALUE_CLOSURE_PTR) {
        uint64_t ptr = tagged.data.ptr_val;
        eshkol_closure_t* closure = (eshkol_closure_t*)ptr;
        return closure ? closure->return_type : CLOSURE_RETURN_UNKNOWN;
    }

    return CLOSURE_RETURN_UNKNOWN;
}

/**
 * @brief Check whether a closure's static return type is a vector/tensor.
 * @param tagged Tagged value expected to hold a closure.
 * @return true if eshkol_closure_get_return_type() reports CLOSURE_RETURN_VECTOR.
 * @note Used by autodiff to pick the scalar vs. tensor differentiation path.
 */
static inline bool eshkol_closure_returns_vector(eshkol_tagged_value_t tagged) {
    return eshkol_closure_get_return_type(tagged) == CLOSURE_RETURN_VECTOR;
}

/**
 * @brief Check whether a closure's static return type is a scalar.
 * @param tagged Tagged value expected to hold a closure.
 * @return true if eshkol_closure_get_return_type() reports CLOSURE_RETURN_SCALAR.
 * @note Used by autodiff to pick the scalar vs. tensor differentiation path.
 */
static inline bool eshkol_closure_returns_scalar(eshkol_tagged_value_t tagged) {
    return eshkol_closure_get_return_type(tagged) == CLOSURE_RETURN_SCALAR;
}


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CLOSURE_H */
