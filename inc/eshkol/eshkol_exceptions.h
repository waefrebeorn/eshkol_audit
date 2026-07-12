#ifndef ESHKOL_EXCEPTIONS_H
#define ESHKOL_EXCEPTIONS_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "eshkol_object.h"

#ifdef __cplusplus
extern "C" {
#endif

// Support for R7RS-compatible exception handling (guard, raise, error)

/**
 * @brief Built-in exception type codes for R7RS-style error handling.
 *
 * Stored in eshkol_exception_t::type; ESHKOL_EXCEPTION_USER_DEFINED marks
 * exceptions raised by user code via `(raise ...)` / `(error ...)` with a
 * condition type not covered by the built-in categories.
 */
typedef enum {
    ESHKOL_EXCEPTION_ERROR = 0,        // Generic error
    ESHKOL_EXCEPTION_TYPE_ERROR,       // Type mismatch
    ESHKOL_EXCEPTION_FILE_ERROR,       // File operation failed
    ESHKOL_EXCEPTION_READ_ERROR,       // Read/parse error
    ESHKOL_EXCEPTION_SYNTAX_ERROR,     // Syntax error
    ESHKOL_EXCEPTION_RANGE_ERROR,      // Index out of bounds
    ESHKOL_EXCEPTION_ARITY_ERROR,      // Wrong number of arguments
    ESHKOL_EXCEPTION_DIVIDE_BY_ZERO,   // Division by zero
    ESHKOL_EXCEPTION_USER_DEFINED      // User-defined exception
} eshkol_exception_type_t;

/**
 * @brief Arena-allocated exception/condition object.
 *
 * Represents a raised R7RS condition: its type code, human-readable
 * message, an array of irritant values (additional data attached via
 * `error`/`raise`), and source location for diagnostics. Wrapped in a
 * tagged value via eshkol_make_exception_value() for use as an ordinary
 * Scheme value (e.g. bound by `guard`).
 */
typedef struct eshkol_exception {
    eshkol_exception_type_t type;      // Exception type code
    char* message;                      // Error message
    eshkol_tagged_value_t* irritants;   // Array of irritant values
    uint32_t num_irritants;             // Number of irritants
    uint32_t line;                      // Source line (0 = unknown)
    uint32_t column;                    // Source column (0 = unknown)
    char* filename;                     // Source filename (NULL = unknown)
} eshkol_exception_t;

/**
 * @brief One entry in the `guard`/exception handler stack.
 *
 * Each active `guard` form pushes one of these, storing the setjmp buffer
 * to longjmp back into when an exception is raised, and linking to the
 * previously active handler so it can be restored on exit.
 */
typedef struct eshkol_exception_handler {
    void* jmp_buf_ptr;                  // Pointer to setjmp buffer
    struct eshkol_exception_handler* prev;  // Previous handler in stack
} eshkol_exception_handler_t;

// Global exception state (thread-local in multi-threaded context)
// Current exception being handled (NULL if none)
extern eshkol_exception_t* g_current_exception;
// Top of exception handler stack (NULL if no handlers)
extern eshkol_exception_handler_t* g_exception_handler_stack;

// Exception API functions (implemented in arena_memory.cpp)
/**
 * @brief Allocate a new exception object of the given type and message.
 * @param type Exception type code.
 * @param message Human-readable error message; copied/owned by the runtime arena.
 * @return Newly allocated eshkol_exception_t, with no irritants and no location set.
 */
eshkol_exception_t* eshkol_make_exception(eshkol_exception_type_t type, const char* message);
/**
 * @brief Like eshkol_make_exception(), but also allocates the object's heap header.
 *
 * Use this variant when the exception will be wrapped directly into an
 * ESHKOL_VALUE_HEAP_PTR-tagged value (subtype HEAP_SUBTYPE_EXCEPTION) rather
 * than the legacy ESHKOL_VALUE_EXCEPTION tag.
 * @param type Exception type code.
 * @param message Human-readable error message.
 * @return Newly allocated eshkol_exception_t preceded by an object header.
 */
eshkol_exception_t* eshkol_make_exception_with_header(eshkol_exception_type_t type, const char* message);
/**
 * @brief Append an irritant value to an exception's irritant list.
 * @param exc Exception to modify.
 * @param irritant Tagged value to append (copied by value).
 */
void eshkol_exception_add_irritant(eshkol_exception_t* exc, eshkol_tagged_value_t irritant);
/**
 * @brief Append an irritant value to an exception's irritant list, by pointer.
 * @param exc Exception to modify.
 * @param irritant Pointer to the tagged value to append (dereferenced and copied).
 */
void eshkol_exception_add_irritant_ptr(eshkol_exception_t* exc, const eshkol_tagged_value_t* irritant);
/**
 * @brief Attach source-location information to an exception for diagnostics.
 * @param exc Exception to modify.
 * @param line Source line (0 = unknown).
 * @param column Source column (0 = unknown).
 * @param filename Source file name (NULL = unknown); ownership per the implementation's convention.
 */
void eshkol_exception_set_location(eshkol_exception_t* exc, uint32_t line, uint32_t column, const char* filename);
/**
 * @brief Raise an exception, transferring control to the innermost active handler.
 *
 * Sets @p exception as `g_current_exception` and longjmps to the buffer at
 * the top of `g_exception_handler_stack`. Does not return if a handler is
 * active; behavior with no active handler follows the runtime's top-level
 * uncaught-exception path.
 * @param exception Exception to raise.
 */
void eshkol_raise(eshkol_exception_t* exception);
// R7RS error-object accessors (implemented in runtime_exceptions_hosted.cpp)
/**
 * @brief R7RS `error-object?` predicate.
 * @param obj Tagged value to test.
 * @return Nonzero if @p obj is an error/exception object, zero otherwise.
 */
int  eshkol_error_object_p(const eshkol_tagged_value_t* obj);
/**
 * @brief R7RS `error-object-message` accessor.
 * @param obj Error object to query.
 * @param out Output tagged value receiving the message string.
 */
void eshkol_error_object_message(const eshkol_tagged_value_t* obj, eshkol_tagged_value_t* out);
/**
 * @brief R7RS `error-object-irritants` accessor.
 * @param obj Error object to query.
 * @param out Output tagged value receiving the list of irritants.
 */
void eshkol_error_object_irritants(const eshkol_tagged_value_t* obj, eshkol_tagged_value_t* out);
/**
 * @brief Push a new exception handler frame onto the handler stack.
 * @param jmp_buf_ptr Pointer to the setjmp buffer to longjmp to on raise.
 */
void eshkol_push_exception_handler(void* jmp_buf_ptr);
/**
 * @brief Pop the innermost exception handler frame, restoring the previous one.
 */
void eshkol_pop_exception_handler(void);
/**
 * @brief Test whether an exception matches a given type code.
 * @param exc Exception to test.
 * @param type Type code to compare against.
 * @return Nonzero if `exc->type == type`, zero otherwise.
 */
int eshkol_exception_type_matches(eshkol_exception_t* exc, eshkol_exception_type_t type);

/**
 * @brief Wrap an exception pointer in a tagged value (legacy EXCEPTION tag).
 * @param exc Exception object to wrap.
 * @return A tagged value of type ESHKOL_VALUE_EXCEPTION pointing at @p exc.
 */
static inline eshkol_tagged_value_t eshkol_make_exception_value(eshkol_exception_t* exc) {
    eshkol_tagged_value_t result;
    result.type = ESHKOL_VALUE_EXCEPTION;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = (uint64_t)exc;
    return result;
}

// ===== END EXCEPTION HANDLING STRUCTURES =====

// ===== FIRST-CLASS CONTINUATIONS =====

/**
 * @brief Captured state for a first-class continuation created by `call/cc`.
 *
 * Holds the setjmp point on the capturing call/cc's stack frame, the value
 * to be delivered when the continuation is later invoked, and the
 * dynamic-wind stack marker at capture time (used to correctly run
 * before/after thunks when the continuation is invoked from a different
 * dynamic extent).
 */
typedef struct eshkol_continuation_state {
    void* jmp_buf_ptr;                  // Points to jmp_buf on the call/cc caller's stack
    eshkol_tagged_value_t value;        // Value passed when continuation is invoked
    void* wind_mark;                    // Dynamic-wind stack marker at capture time
} eshkol_continuation_state_t;

/**
 * @brief One entry in the `dynamic-wind` stack.
 *
 * Records the before/after thunks of an active `dynamic-wind` so that
 * eshkol_unwind_dynamic_wind() can run the appropriate after-thunks when a
 * continuation invocation unwinds past this dynamic extent.
 */
typedef struct eshkol_dynamic_wind_entry {
    eshkol_tagged_value_t before;       // Before thunk (callable)
    eshkol_tagged_value_t after;        // After thunk (callable)
    struct eshkol_dynamic_wind_entry* prev;  // Previous entry in stack
} eshkol_dynamic_wind_entry_t;

// Global dynamic-wind stack
extern eshkol_dynamic_wind_entry_t* g_dynamic_wind_stack;

// Continuation runtime functions
/**
 * @brief Allocate the captured state for a new continuation.
 * @param arena Arena to allocate from.
 * @param jmp_buf_ptr setjmp buffer captured at the call/cc call site.
 * @return Newly allocated eshkol_continuation_state_t.
 */
eshkol_continuation_state_t* eshkol_make_continuation_state(void* arena, void* jmp_buf_ptr);
/**
 * @brief Wrap a captured continuation state in an invocable closure.
 * @param arena Arena to allocate the closure from.
 * @param state_ptr Continuation state previously created by
 *        eshkol_make_continuation_state(); invoking the returned closure
 *        with a value longjmps back to the capture point delivering it.
 * @return Opaque pointer to the resulting closure object.
 */
void* eshkol_make_continuation_closure(void* arena, void* state_ptr);
/**
 * @brief Push a new dynamic-wind frame with the given before/after thunks.
 * @param arena Arena to allocate the stack entry from.
 * @param before Thunk invoked when entering this dynamic extent.
 * @param after Thunk invoked when leaving this dynamic extent.
 */
void eshkol_push_dynamic_wind(void* arena, const eshkol_tagged_value_t* before, const eshkol_tagged_value_t* after);
/**
 * @brief Pop the innermost dynamic-wind frame without running its after-thunk.
 *
 * Used for normal (non-continuation) exit from the dynamic-wind body, where
 * the after-thunk is invoked directly by the generated code.
 */
void eshkol_pop_dynamic_wind(void);
/**
 * @brief Run after-thunks for all dynamic-wind frames above a saved marker.
 *
 * Invoked when a continuation is called from outside the dynamic extent in
 * which it was captured, to correctly unwind (running after-thunks) or
 * rewind (running before-thunks) through intervening dynamic-wind frames.
 * @param saved_wind_mark Dynamic-wind stack marker captured with the
 *        continuation (eshkol_continuation_state_t::wind_mark).
 */
void eshkol_unwind_dynamic_wind(void* saved_wind_mark);

// ===== END FIRST-CLASS CONTINUATIONS =====

/**
 * @brief Initialize process/thread stack sizing for deep recursion support.
 *
 * Should be called early in program startup (before deep recursive Scheme
 * code runs) to raise the OS thread/process stack limit so that
 * non-tail-recursive Scheme programs have adequate headroom before
 * overflowing the native stack.
 */
void eshkol_init_stack_size(void);


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_EXCEPTIONS_H */
