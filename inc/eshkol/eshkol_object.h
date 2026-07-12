#ifndef ESHKOL_OBJECT_H
#define ESHKOL_OBJECT_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "eshkol_value.h"
#include "eshkol_typecheck.h"

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════
// OBJECT HEADER SYSTEM (Foundation for Pointer Consolidation)
// All heap-allocated objects will be prefixed with this header.
// The header enables type consolidation: multiple specific types (cons, string,
// vector, etc.) are unified under HEAP_PTR with subtypes stored in the header.
// ═══════════════════════════════════════════════════════════════════════════

// Object flags for GC, linear types, and resource management
#define ESHKOL_OBJ_FLAG_MARKED    0x01  // GC mark bit
#define ESHKOL_OBJ_FLAG_LINEAR    0x02  // Linear type (must be consumed exactly once)
#define ESHKOL_OBJ_FLAG_BORROWED  0x04  // Currently borrowed (temporary access)
#define ESHKOL_OBJ_FLAG_CONSUMED  0x08  // Linear value has been consumed
#define ESHKOL_OBJ_FLAG_SHARED    0x10  // Reference-counted shared object
#define ESHKOL_OBJ_FLAG_WEAK      0x20  // Weak reference (doesn't prevent collection)
#define ESHKOL_OBJ_FLAG_PINNED    0x40  // Pinned in memory (no relocation)
#define ESHKOL_OBJ_FLAG_EXTERNAL  0x80  // External resource (needs explicit cleanup)

/**
 * @brief 8-byte header prepended to every heap-allocated object.
 *
 * Enables pointer-type consolidation: ESHKOL_VALUE_HEAP_PTR, CALLABLE,
 * HANDLE, BUFFER, STREAM, and EVENT tagged values all point past one of
 * these headers, with `subtype` identifying the concrete representation
 * (see heap_subtype_t / callable_subtype_t / handle_subtype_t / etc.).
 * Accessed via the ESHKOL_GET_HEADER family of macros below.
 */
typedef struct eshkol_object_header {
    uint8_t  subtype;      // Distinguishes types within HEAP_PTR/CALLABLE/HANDLE
    uint8_t  flags;        // Object flags (GC marks, linear status, etc.)
    uint16_t ref_count;    // Reference count for shared objects (0 = not ref-counted)
    uint32_t size;         // Object size in bytes (excluding header)
} eshkol_object_header_t;

// Compile-time validation
ESHKOL_STATIC_ASSERT(sizeof(eshkol_object_header_t) == 8,
                     "Object header must be 8 bytes for alignment");

// ───────────────────────────────────────────────────────────────────────────
// HEAP_PTR SUBTYPES (type = ESHKOL_VALUE_HEAP_PTR = 8)
// Data structures allocated on the arena
// ───────────────────────────────────────────────────────────────────────────
/**
 * @brief Concrete data-structure kind for a value tagged ESHKOL_VALUE_HEAP_PTR.
 *
 * Stored in the eshkol_object_header_t::subtype field prepended to the
 * object. Distinguishes the arena-allocated heap data structures (lists,
 * strings, vectors, tensors, hash tables, records, ports, bignums, and the
 * neuro-symbolic/AD structures added for the consciousness engine and
 * Taylor-tower AD) that share the single consolidated HEAP_PTR type tag.
 */
typedef enum {
    HEAP_SUBTYPE_CONS        = 0,   // Cons cell (pair/list node)
    HEAP_SUBTYPE_STRING      = 1,   // String (UTF-8 with length)
    HEAP_SUBTYPE_VECTOR      = 2,   // Heterogeneous vector
    HEAP_SUBTYPE_TENSOR      = 3,   // N-dimensional numeric tensor
    HEAP_SUBTYPE_MULTI_VALUE = 4,   // Multiple return values container
    HEAP_SUBTYPE_HASH        = 5,   // Hash table / dictionary
    HEAP_SUBTYPE_EXCEPTION   = 6,   // Exception object
    HEAP_SUBTYPE_RECORD      = 7,   // User-defined record type
    HEAP_SUBTYPE_BYTEVECTOR  = 8,   // Raw byte vector (R7RS)
    HEAP_SUBTYPE_PORT        = 9,   // I/O port
    HEAP_SUBTYPE_SYMBOL      = 10,  // Interned symbol (distinct from string)
    HEAP_SUBTYPE_BIGNUM      = 11,  // Arbitrary-precision integer (R7RS exact)
    // Neuro-symbolic consciousness engine types
    HEAP_SUBTYPE_SUBSTITUTION    = 12,  // Immutable binding map {var_id -> tagged_value}
    HEAP_SUBTYPE_FACT            = 13,  // Predicate + arguments: (pred arg1 arg2 ...)
    // Reserved: 14 for RULE (v1.2 backward chaining)
    HEAP_SUBTYPE_KNOWLEDGE_BASE  = 15,  // Collection of facts with query support
    HEAP_SUBTYPE_FACTOR_GRAPH    = 16,  // Factor graph for probabilistic inference
    HEAP_SUBTYPE_WORKSPACE       = 17,  // Global workspace for cognitive competition
    HEAP_SUBTYPE_PROMISE         = 18,  // Lazy promise (delay/force with memoization)
    HEAP_SUBTYPE_RATIONAL        = 19,  // Exact rational number (numerator/denominator)
    HEAP_SUBTYPE_PRNG            = 20,  // Isolated pseudo-random number generator state
    HEAP_SUBTYPE_DNC             = 21,  // Differentiable external memory (NTM/DNC head)
    HEAP_SUBTYPE_SDNC            = 22,  // SDNC weight-program handle (bytecode-VM-as-transformer θ)
    HEAP_SUBTYPE_TAYLOR          = 23,  // Truncated-Taylor tower for arbitrary-order AD (ESH-0186)
    HEAP_SUBTYPE_PARAMETER       = 24,  // R7RS dynamic parameter object (make-parameter/parameterize)
    // Reserved: 25-255 for future heap types
} heap_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// CALLABLE SUBTYPES (type = ESHKOL_VALUE_CALLABLE = 9)
// Function-like objects that can be invoked
// ───────────────────────────────────────────────────────────────────────────
/**
 * @brief Concrete callable kind for a value tagged ESHKOL_VALUE_CALLABLE.
 *
 * Stored in the eshkol_object_header_t::subtype field prepended to the
 * callable object, distinguishing compiled closures, homoiconic
 * lambda-as-data representations, autodiff nodes, built-in primitives, and
 * (reserved for future use) first-class continuations.
 */
typedef enum {
    CALLABLE_SUBTYPE_CLOSURE      = 0,  // Compiled closure (func_ptr + env)
    CALLABLE_SUBTYPE_LAMBDA_SEXPR = 1,  // Lambda as data (homoiconicity)
    CALLABLE_SUBTYPE_AD_NODE      = 2,  // Autodiff computation node
    CALLABLE_SUBTYPE_PRIMITIVE    = 3,  // Built-in primitive function
    CALLABLE_SUBTYPE_CONTINUATION = 4,  // First-class continuation (future)
    // Reserved: 5-255 for future callable types
} callable_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// HANDLE SUBTYPES (type = ESHKOL_VALUE_HANDLE = 16, future multimedia)
// External resources requiring explicit lifecycle management
// ───────────────────────────────────────────────────────────────────────────
/**
 * @brief Concrete external-resource kind for a value tagged ESHKOL_VALUE_HANDLE.
 *
 * Reserved for the future multimedia/systems API: window, GPU context,
 * audio/MIDI/camera devices, sockets, framebuffers, files, threads, and
 * mutexes each requiring explicit lifecycle management outside the GC.
 */
typedef enum {
    HANDLE_SUBTYPE_WINDOW       = 0,   // Window/surface handle
    HANDLE_SUBTYPE_GL_CONTEXT   = 1,   // OpenGL/Vulkan/Metal context
    HANDLE_SUBTYPE_AUDIO_DEVICE = 2,   // Audio output device
    HANDLE_SUBTYPE_MIDI_PORT    = 3,   // MIDI I/O port
    HANDLE_SUBTYPE_CAMERA       = 4,   // Camera capture device
    HANDLE_SUBTYPE_SOCKET       = 5,   // Network socket
    HANDLE_SUBTYPE_FRAMEBUFFER  = 6,   // Offscreen render target
    HANDLE_SUBTYPE_FILE         = 7,   // File handle (distinct from PORT)
    HANDLE_SUBTYPE_THREAD       = 8,   // Thread handle
    HANDLE_SUBTYPE_MUTEX        = 9,   // Mutex/lock handle
    // Reserved: 10-255 for future handle types
} handle_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// BUFFER SUBTYPES (type = ESHKOL_VALUE_BUFFER = 17, future multimedia)
// Memory regions for zero-copy data transfer
// ───────────────────────────────────────────────────────────────────────────
/**
 * @brief Concrete memory-region kind for a value tagged ESHKOL_VALUE_BUFFER.
 *
 * Reserved for the future multimedia API's zero-copy data buffers (raw
 * bytes, image/audio samples, GPU vertex/index/uniform buffers, textures,
 * and memory-mapped files).
 */
typedef enum {
    BUFFER_SUBTYPE_RAW         = 0,   // Raw byte buffer
    BUFFER_SUBTYPE_IMAGE       = 1,   // Image pixel data
    BUFFER_SUBTYPE_AUDIO       = 2,   // Audio sample data
    BUFFER_SUBTYPE_VERTEX      = 3,   // GPU vertex buffer
    BUFFER_SUBTYPE_INDEX       = 4,   // GPU index buffer
    BUFFER_SUBTYPE_UNIFORM     = 5,   // GPU uniform buffer
    BUFFER_SUBTYPE_TEXTURE     = 6,   // Texture data
    BUFFER_SUBTYPE_MAPPED      = 7,   // Memory-mapped file
    // Reserved: 8-255 for future buffer types
} buffer_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// STREAM SUBTYPES (type = ESHKOL_VALUE_STREAM = 18, future multimedia)
// Async I/O and data flow pipelines
// ───────────────────────────────────────────────────────────────────────────
/**
 * @brief Concrete async data-flow kind for a value tagged ESHKOL_VALUE_STREAM.
 *
 * Reserved for the future multimedia API's lazy/async I/O pipelines (byte,
 * audio, video, MIDI, and network streams, plus transform/filter stages).
 */
typedef enum {
    STREAM_SUBTYPE_BYTE        = 0,   // Raw byte stream
    STREAM_SUBTYPE_AUDIO       = 1,   // Audio sample stream
    STREAM_SUBTYPE_VIDEO       = 2,   // Video frame stream
    STREAM_SUBTYPE_MIDI        = 3,   // MIDI event stream
    STREAM_SUBTYPE_NETWORK     = 4,   // Network data stream
    STREAM_SUBTYPE_TRANSFORM   = 5,   // Transform/filter pipeline
    // Reserved: 6-255 for future stream types
} stream_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// EVENT SUBTYPES (type = ESHKOL_VALUE_EVENT = 19, future multimedia)
// Real-time event handling
// ───────────────────────────────────────────────────────────────────────────
/**
 * @brief Concrete real-time event kind for a value tagged ESHKOL_VALUE_EVENT.
 *
 * Reserved for the future multimedia API's event system (input, window,
 * audio, MIDI, timer, network, and user-defined custom events).
 */
typedef enum {
    EVENT_SUBTYPE_INPUT        = 0,   // Keyboard/mouse/touch input
    EVENT_SUBTYPE_WINDOW       = 1,   // Window resize/focus/close
    EVENT_SUBTYPE_AUDIO        = 2,   // Audio buffer ready/underrun
    EVENT_SUBTYPE_MIDI         = 3,   // MIDI note/CC/sysex
    EVENT_SUBTYPE_TIMER        = 4,   // Timer tick
    EVENT_SUBTYPE_NETWORK      = 5,   // Network packet/connection
    EVENT_SUBTYPE_CUSTOM       = 6,   // User-defined event
    // Reserved: 7-255 for future event types
} event_subtype_t;

// ───────────────────────────────────────────────────────────────────────────
// HEADER ACCESS MACROS
// These macros allow accessing the header from a data pointer
// Memory layout: [header (8 bytes)][object data (variable)]
// ───────────────────────────────────────────────────────────────────────────

// Get header pointer from data pointer (subtracts header size)
#define ESHKOL_GET_HEADER(data_ptr) \
    ((eshkol_object_header_t*)((uint8_t*)(data_ptr) - sizeof(eshkol_object_header_t)))

// Get data pointer from header pointer (adds header size)
#define ESHKOL_GET_DATA_PTR(header_ptr) \
    ((void*)((uint8_t*)(header_ptr) + sizeof(eshkol_object_header_t)))

// Get subtype from data pointer
#define ESHKOL_GET_SUBTYPE(data_ptr) \
    (ESHKOL_GET_HEADER(data_ptr)->subtype)

// Get flags from data pointer
#define ESHKOL_GET_FLAGS(data_ptr) \
    (ESHKOL_GET_HEADER(data_ptr)->flags)

// Set subtype on data pointer
#define ESHKOL_SET_SUBTYPE(data_ptr, st) \
    (ESHKOL_GET_HEADER(data_ptr)->subtype = (st))

// Set flags on data pointer
#define ESHKOL_SET_FLAGS(data_ptr, fl) \
    (ESHKOL_GET_HEADER(data_ptr)->flags = (fl))

// Check if object has specific flag
#define ESHKOL_HAS_FLAG(data_ptr, flag) \
    ((ESHKOL_GET_FLAGS(data_ptr) & (flag)) != 0)

// Set a specific flag
#define ESHKOL_ADD_FLAG(data_ptr, flag) \
    (ESHKOL_GET_HEADER(data_ptr)->flags |= (flag))

// Clear a specific flag
#define ESHKOL_CLEAR_FLAG(data_ptr, flag) \
    (ESHKOL_GET_HEADER(data_ptr)->flags &= ~(flag))

// Get object size from data pointer
#define ESHKOL_GET_OBJ_SIZE(data_ptr) \
    (ESHKOL_GET_HEADER(data_ptr)->size)

// Get reference count from data pointer
#define ESHKOL_GET_REF_COUNT(data_ptr) \
    (ESHKOL_GET_HEADER(data_ptr)->ref_count)

// Increment reference count (returns new count)
#define ESHKOL_INC_REF(data_ptr) \
    (++(ESHKOL_GET_HEADER(data_ptr)->ref_count))

// Decrement reference count (returns new count)
#define ESHKOL_DEC_REF(data_ptr) \
    (--(ESHKOL_GET_HEADER(data_ptr)->ref_count))

// ───────────────────────────────────────────────────────────────────────────
// COMPATIBILITY MACROS FOR TYPE CHECKING (Display System Use)
// ───────────────────────────────────────────────────────────────────────────
// These macros support BOTH legacy and consolidated formats for the display
// system to render old data. M1 Migration is COMPLETE - new code should use:
//   - HEAP_PTR type + ESHKOL_GET_SUBTYPE() to check heap subtypes
//   - CALLABLE type + ESHKOL_GET_SUBTYPE() to check callable subtypes
// ───────────────────────────────────────────────────────────────────────────

// Check if value is a cons cell (legacy CONS_PTR or new HEAP_PTR with CONS subtype)
#define ESHKOL_IS_CONS_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_CONS_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_CONS))

// Check if value is a string (legacy STRING_PTR or new HEAP_PTR with STRING subtype)
#define ESHKOL_IS_STRING_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_STRING_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_STRING))

// Check if value is a vector (legacy VECTOR_PTR or new HEAP_PTR with VECTOR subtype)
#define ESHKOL_IS_VECTOR_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_VECTOR_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_VECTOR))

// Check if value is a tensor (legacy TENSOR_PTR or new HEAP_PTR with TENSOR subtype)
#define ESHKOL_IS_TENSOR_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_TENSOR_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_TENSOR))

// Check if value is a hash table (legacy HASH_PTR or new HEAP_PTR with HASH subtype)
#define ESHKOL_IS_HASH_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_HASH_PTR || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_HASH))

// Check if value is an exception (legacy EXCEPTION or new HEAP_PTR with EXCEPTION subtype)
#define ESHKOL_IS_EXCEPTION_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_EXCEPTION || \
     ((val).type == ESHKOL_VALUE_HEAP_PTR && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_EXCEPTION))

// Check if value is a multi-value (ONLY new HEAP_PTR with MULTI_VALUE subtype)
#define ESHKOL_IS_MULTI_VALUE(val) \
    ((val).type == ESHKOL_VALUE_HEAP_PTR && \
     (val).data.ptr_val != 0 && \
     ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_MULTI_VALUE)

// Check if value is a bignum
#define ESHKOL_IS_BIGNUM(val) \
    ((val).type == ESHKOL_VALUE_HEAP_PTR && \
     (val).data.ptr_val != 0 && \
     ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == HEAP_SUBTYPE_BIGNUM)

// Check if value is a closure (legacy CLOSURE_PTR or new CALLABLE with CLOSURE subtype)
#define ESHKOL_IS_CLOSURE_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_CLOSURE_PTR || \
     ((val).type == ESHKOL_VALUE_CALLABLE && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == CALLABLE_SUBTYPE_CLOSURE))

// Check if value is a lambda sexpr (legacy LAMBDA_SEXPR or new CALLABLE with LAMBDA_SEXPR subtype)
#define ESHKOL_IS_LAMBDA_SEXPR_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_LAMBDA_SEXPR || \
     ((val).type == ESHKOL_VALUE_CALLABLE && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == CALLABLE_SUBTYPE_LAMBDA_SEXPR))

// Check if value is an AD node (legacy AD_NODE_PTR or new CALLABLE with AD_NODE subtype)
#define ESHKOL_IS_AD_NODE_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_AD_NODE_PTR || \
     ((val).type == ESHKOL_VALUE_CALLABLE && \
      (val).data.ptr_val != 0 && \
      ESHKOL_GET_SUBTYPE((void*)(val).data.ptr_val) == CALLABLE_SUBTYPE_AD_NODE))

// Check if value is any heap pointer (legacy individual types OR new consolidated HEAP_PTR)
#define ESHKOL_IS_HEAP_PTR_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_HEAP_PTR || \
     (val).type == ESHKOL_VALUE_CONS_PTR || \
     (val).type == ESHKOL_VALUE_STRING_PTR || \
     (val).type == ESHKOL_VALUE_VECTOR_PTR || \
     (val).type == ESHKOL_VALUE_TENSOR_PTR || \
     (val).type == ESHKOL_VALUE_HASH_PTR || \
     (val).type == ESHKOL_VALUE_EXCEPTION)

// Check if value is any callable (legacy individual types OR new consolidated CALLABLE)
#define ESHKOL_IS_CALLABLE_COMPAT(val) \
    ((val).type == ESHKOL_VALUE_CALLABLE || \
     (val).type == ESHKOL_VALUE_CLOSURE_PTR || \
     (val).type == ESHKOL_VALUE_LAMBDA_SEXPR || \
     (val).type == ESHKOL_VALUE_AD_NODE_PTR)

// Check if value is a multimedia handle
#define ESHKOL_IS_HANDLE(val) ((val).type == ESHKOL_VALUE_HANDLE)

// Check if value is a multimedia buffer
#define ESHKOL_IS_BUFFER(val) ((val).type == ESHKOL_VALUE_BUFFER)

// Check if value is a multimedia stream
#define ESHKOL_IS_STREAM(val) ((val).type == ESHKOL_VALUE_STREAM)

// Check if value is a multimedia event
#define ESHKOL_IS_EVENT(val)  ((val).type == ESHKOL_VALUE_EVENT)


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_OBJECT_H */
