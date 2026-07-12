#ifndef ESHKOL_VERSION_H
#define ESHKOL_VERSION_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESHKOL_VERSION_MAJOR 1
#define ESHKOL_VERSION_MINOR 3
#define ESHKOL_VERSION_PATCH 3
#define ESHKOL_VERSION_STRING "1.3.3-evolve"

#include <stdint.h>
#include <stdbool.h>

// ═══════════════════════════════════════════════════════════════════════════
// CROSS-PLATFORM STATIC ASSERTION MACRO
// _Static_assert is C11, static_assert is C++11. GCC in C++ mode doesn't
// recognize _Static_assert. This macro provides a unified interface.
// ═══════════════════════════════════════════════════════════════════════════
/**
 * @brief Compile-time assertion, portable across C11 and C++11 compilation.
 *
 * Expands to `static_assert` in C++ and `_Static_assert` in C, since GCC in
 * C++ mode does not recognize `_Static_assert`. @p cond must be a constant
 * expression; @p msg is the diagnostic emitted if it evaluates to false.
 */
#ifdef __cplusplus
#define ESHKOL_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ESHKOL_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif


#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_VERSION_H */
