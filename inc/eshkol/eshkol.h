/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * @file eshkol.h
 * @brief Aggregator header for the Eshkol runtime/compiler public API.
 *
 * This file is a thin umbrella over the focused sub-headers in inc/eshkol/.
 * It exists so legacy `#include "eshkol/eshkol.h"` sites keep working; new code
 * should include the specific sub-header it needs (no-god-header discipline):
 *
 *   eshkol_version.h      - version numbers + portable static-assert macro
 *   eshkol_type.h         - eshkol_type_t (frontend/AST value kinds)
 *   eshkol_value.h        - eshkol_value_type_t runtime tag + flags
 *   eshkol_typecheck.h    - type-checking helper macros
 *   eshkol_object.h       - object header system (pointer consolidation)
 *   eshkol_ad.h           - dual numbers + reverse-mode AD nodes/tape
 *   eshkol_closure.h      - closures + primitive callables
 *   eshkol_exceptions.h   - exception system + continuations + dynamic-wind
 *   eshkol_runtime.h      - lambda registry, display, process globals
 *   eshkol_hott.h         - HoTT-inspired static type expressions
 *   eshkol_macro.h        - syntax-rules (macro) definitions
 *   eshkol_ast.h          - operation + AST node definitions
 */
#ifndef ESHKOL_ESHKOL_H
#define ESHKOL_ESHKOL_H

#include "eshkol_version.h"
#include "eshkol_type.h"
#include "eshkol_value.h"
#include "eshkol_typecheck.h"
#include "eshkol_object.h"
#include "eshkol_ad.h"
#include "eshkol_closure.h"
#include "eshkol_exceptions.h"
#include "eshkol_runtime.h"
#include "eshkol_hott.h"
#include "eshkol_macro.h"
#include "eshkol_ast.h"

#endif /* ESHKOL_ESHKOL_H */
