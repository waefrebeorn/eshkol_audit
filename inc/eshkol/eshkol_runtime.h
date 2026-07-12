#ifndef ESHKOL_RUNTIME_H
#define ESHKOL_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "eshkol_closure.h"
#include "eshkol_exceptions.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runtime table mapping function pointers to their S-expression representations
// This enables full homoiconicity: (display (list double)) shows the lambda source

/**
 * @brief One entry mapping a compiled function pointer to its source form.
 */
typedef struct eshkol_lambda_entry {
    uint64_t func_ptr;      // Function pointer as uint64
    uint64_t sexpr_ptr;     // Pointer to S-expression cons cell (0 if none)
    const char* name;       // Function name for debugging (may be NULL)
} eshkol_lambda_entry_t;

/**
 * @brief Growable table of eshkol_lambda_entry_t used for homoiconicity.
 */
typedef struct eshkol_lambda_registry {
    eshkol_lambda_entry_t* entries;
    size_t count;
    size_t capacity;
} eshkol_lambda_registry_t;

// Global lambda registry (defined in arena_memory.cpp)
extern eshkol_lambda_registry_t* g_lambda_registry;

// Lambda registry API
/**
 * @brief Initialize the global lambda registry (allocate backing storage).
 *
 * Must be called once during runtime startup before any
 * eshkol_lambda_registry_add() calls.
 */
void eshkol_lambda_registry_init(void);
/**
 * @brief Free the global lambda registry's backing storage.
 */
void eshkol_lambda_registry_destroy(void);
/**
 * @brief Register a compiled function pointer's S-expression source and name.
 * @param func_ptr Compiled function pointer, as a uint64.
 * @param sexpr_ptr Pointer to the cons-cell S-expression representation (0 if none).
 * @param name Function name for debugging/display (may be NULL for anonymous lambdas).
 */
void eshkol_lambda_registry_add(uint64_t func_ptr, uint64_t sexpr_ptr, const char* name);
/**
 * @brief Look up the S-expression pointer registered for a function pointer.
 * @param func_ptr Compiled function pointer to look up.
 * @return The registered sexpr_ptr, or 0 if @p func_ptr is not registered.
 */
uint64_t eshkol_lambda_registry_lookup(uint64_t func_ptr);

// ===== END LAMBDA REGISTRY =====

// ===== UNIFIED DISPLAY SYSTEM =====
// Single source of truth for displaying all Eshkol values

// Forward declaration for tagged cons cell
struct arena_tagged_cons_cell;

/**
 * @brief Options controlling how eshkol_display_value_opts() renders a value.
 */
typedef struct eshkol_display_opts {
    int max_depth;          // Maximum recursion depth (default: 100)
    int current_depth;      // Current depth (internal use)
    uint8_t quote_strings;  // Quote strings with "" (true for 'write', false for 'display')
    uint8_t show_types;     // Debug: show type tags
    void* output;           // Output stream (FILE*, default: stdout)
} eshkol_display_opts_t;

/**
 * @brief Construct the default eshkol_display_opts_t (display semantics, stdout).
 * @return Options with max_depth=100, unquoted strings, no type tags, output=stdout.
 */
static inline eshkol_display_opts_t eshkol_display_default_opts(void) {
    eshkol_display_opts_t opts;
    opts.max_depth = 100;
    opts.current_depth = 0;
    opts.quote_strings = 0;
    opts.show_types = 0;
    opts.output = 0;  // NULL means stdout
    return opts;
}

// R7RS flonum external representation (+inf.0 / -inf.0 / +nan.0). Single
// source of truth shared by display/write/number->string/logic printing.
// Defined in runtime_display_hosted.cpp.
/**
 * @brief Format a double using Eshkol's R7RS flonum external representation.
 * @param buf Destination buffer.
 * @param n Size of @p buf in bytes.
 * @param v Value to format.
 * @return Number of characters that would be written (as with snprintf),
 *         not counting the terminating NUL; truncated to fit @p n.
 */
int  eshkol_format_double(char* buf, size_t n, double v);
/**
 * @brief Print a double to a FILE* using Eshkol's flonum external representation.
 * @param file Destination stream, as a `FILE*` cast to `void*`.
 * @param v Value to print.
 */
void eshkol_fprint_double(void* file, double v);

// Main display functions (implemented in arena_memory.cpp)
/**
 * @brief Display a tagged value to stdout using default options ('display' semantics).
 * @param value Value to display.
 */
void eshkol_display_value(const eshkol_tagged_value_t* value);
/**
 * @brief Display a tagged value to stdout with caller-specified options.
 * @param value Value to display.
 * @param opts Display options (depth limit, quoting, type-tag debug output, output stream).
 */
void eshkol_display_value_opts(const eshkol_tagged_value_t* value, eshkol_display_opts_t* opts);
void eshkol_write_value(const eshkol_tagged_value_t* value);  // Scheme 'write' semantics
/**
 * @brief Write a tagged value (R7RS `write` semantics: strings/chars quoted) to a port.
 * @param value Value to write.
 * @param port Destination port, as a runtime port pointer cast to `void*`.
 */
void eshkol_write_value_to_port(const eshkol_tagged_value_t* value, void* port);
/**
 * @brief Display a tagged value (R7RS `display` semantics: strings/chars unquoted) to a port.
 * @param value Value to display.
 * @param port Destination port, as a runtime port pointer cast to `void*`.
 */
void eshkol_display_value_to_port(const eshkol_tagged_value_t* value, void* port);

// ─── R7RS current-{input,output,error}-port parameter cells ───────────────
// `display`/`write`/`newline` (no explicit port arg) and the codegen for
// `(current-output-port)` consult these. `(parameterize ((current-output-port
// p)) …)` invokes the setters via the Scheme-level closure that wraps the
// FILE* extracted from the port tagged value.
/** @brief Get the current-output-port's underlying `FILE*` (as `void*`). */
void* eshkol_runtime_current_output_fp(void);
/** @brief Get the current-input-port's underlying `FILE*` (as `void*`). */
void* eshkol_runtime_current_input_fp(void);
/** @brief Get the current-error-port's underlying `FILE*` (as `void*`). */
void* eshkol_runtime_current_error_fp(void);
/**
 * @brief Set the current-output-port's underlying `FILE*`.
 * @param fp New `FILE*`, as `void*`; invoked by `parameterize` on `current-output-port`.
 */
void  eshkol_runtime_set_current_output_fp(void* fp);
/**
 * @brief Set the current-input-port's underlying `FILE*`.
 * @param fp New `FILE*`, as `void*`; invoked by `parameterize` on `current-input-port`.
 */
void  eshkol_runtime_set_current_input_fp(void* fp);
/**
 * @brief Set the current-error-port's underlying `FILE*`.
 * @param fp New `FILE*`, as `void*`; invoked by `parameterize` on `current-error-port`.
 */
void  eshkol_runtime_set_current_error_fp(void* fp);

// Process-global runtime helpers emitted by codegen/JIT modules.
/**
 * @brief Look up a compiled function or global's address by interned symbol name.
 * @param name Symbol name to resolve.
 * @return Address of the resolved symbol, or NULL if not found.
 */
void* eshkol_intern_symbol_lookup(const char* name);
/**
 * @brief Resolve a forward-declared function reference at first call.
 *
 * Emitted by codegen at call sites of functions that may not yet be defined
 * (e.g. mutual recursion in the REPL/JIT). If @p loaded_fn_ptr is non-NULL
 * it is used directly; otherwise the runtime attempts to resolve @p name
 * and reports an error referencing it if resolution fails.
 * @param loaded_fn_ptr Already-resolved function pointer, or NULL if not yet loaded.
 * @param unresolved_stub_ptr Stub pointer to compare against/replace once resolved.
 * @param name Function name, for lookup and diagnostics.
 * @return The resolved function pointer to call.
 */
void* eshkol_check_forward_ref(void* loaded_fn_ptr,
                               void* unresolved_stub_ptr,
                               const char* name);
/**
 * @brief Raise a runtime "not a pair" error for an operation requiring a pair.
 * @param op_name Name of the operation being performed (e.g. "car", "cdr"), used in the error message.
 */
void  eshkol_raise_not_pair(const char* op_name);

/**
 * @brief Display a list (cons cell chain) as its printed representation.
 * @param cons_ptr Pointer to the head cons cell, as a uint64.
 * @param opts Display options.
 */
void eshkol_display_list(uint64_t cons_ptr, eshkol_display_opts_t* opts);

/**
 * @brief Display a raw lambda by looking up its S-expression in the lambda registry.
 * @param func_ptr Compiled function pointer, as a uint64.
 * @param opts Display options.
 */
void eshkol_display_lambda(uint64_t func_ptr, eshkol_display_opts_t* opts);

/**
 * @brief Display a closure by extracting and printing its embedded S-expression.
 * @param closure_ptr Pointer to an eshkol_closure_t, as a uint64.
 * @param opts Display options.
 */
void eshkol_display_closure(uint64_t closure_ptr, eshkol_display_opts_t* opts);

// ===== END UNIFIED DISPLAY SYSTEM =====

// ===== END COMPUTATIONAL GRAPH TYPES =====

// ===== HoTT TYPE SYSTEM =====


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_RUNTIME_H */
