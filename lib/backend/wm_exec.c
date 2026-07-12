static void exec_loop_postprocess(float x[D], const Instr* prog, int n_instr) {
    /* IS_NATIVE: DIV, MOD, etc. */
    if (x[S_IS_NATIVE] > 0.5f) {
        int pc = (int)roundf(x[S_PC]) - 1;
        if (pc >= 0 && pc < n_instr) {
            int opcode = prog[pc].op;
            float tos = x[S_TOS], sos = x[S_SOS];
            float r2 = x[S_R2], r3 = x[S_R3];
            if (opcode == OP_DIV) {
                if (tos == 0) {
                    x[S_HALT] = 1;
                } else {
                    x[S_TOS] = sos / tos;
                    x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                    x[S_DEPTH] -= 1;
                    x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                }
            } else if (opcode == OP_MOD) {
                if (tos == 0) {
                    x[S_HALT] = 1;
                } else {
                    float r = fmodf(sos, tos);
                    if (r != 0 && ((r > 0) != (tos > 0))) r += tos;
                    x[S_TOS] = r;
                    x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                    x[S_DEPTH] -= 1;
                    x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                }
            } else if (opcode == OP_CONS) {
                /* CONS: allocate pair on heap. TOS=cdr, SOS=car → push heap ptr */
                if (g_heap_ptr + 2 <= HEAP_SIZE) {
                    int ptr = g_heap_ptr;
                    g_heap[g_heap_ptr++] = sos;  /* car */
                    g_heap[g_heap_ptr++] = tos;  /* cdr */
                    x[S_TOS] = (float)ptr;       /* pair reference = heap index */
                    x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                    x[S_DEPTH] -= 1;
                    x[S_TYPE_TOS] = TYPE_PAIR; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                }
            } else if (opcode == OP_CAR) {
                /* Type check: accept TYPE_PAIR or valid-looking heap pointer
                 * (simulated/matrix paths may not set type tags for CONS) */
                int ptr = (int)tos;
                int is_pair = (x[S_TYPE_TOS] == TYPE_PAIR) ||
                              (ptr >= 0 && ptr + 1 < g_heap_ptr);
                if (!is_pair) {
                    x[S_TOS] = 0; x[S_TYPE_TOS] = TYPE_NUMBER;
                } else {
                    if (ptr >= 0 && ptr + 1 < HEAP_SIZE)
                        x[S_TOS] = g_heap[ptr];      /* car */
                    x[S_TYPE_TOS] = TYPE_NUMBER; /* element type unknown */
                }
            } else if (opcode == OP_CDR) {
                int ptr = (int)tos;
                int is_pair = (x[S_TYPE_TOS] == TYPE_PAIR) ||
                              (ptr >= 0 && ptr + 1 < g_heap_ptr);
                if (!is_pair) {
                    x[S_TOS] = 0; x[S_TYPE_TOS] = TYPE_NUMBER;
                } else {
                    if (ptr >= 0 && ptr + 1 < HEAP_SIZE)
                        x[S_TOS] = g_heap[ptr + 1];  /* cdr */
                    x[S_TYPE_TOS] = TYPE_NUMBER; /* element type unknown */
                }
            } else if (opcode == OP_NULL_P) {
                /* NULL_P: check both type tag and value sentinel for compatibility
                 * with all three execution paths (ref sets type, sim/mat use value) */
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_NIL || tos == -1.0f) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_SET_CAR) {
                /* SET_CAR: TOS=val, SOS=pair → mutate car */
                int ptr = (int)sos;
                if (ptr >= 0 && ptr < HEAP_SIZE) g_heap[ptr] = tos;
                x[S_TOS] = r2; x[S_SOS] = r3; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 2;
                x[S_TYPE_TOS] = x[S_TYPE_R2]; x[S_TYPE_SOS] = x[S_TYPE_R3]; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_SET_CDR) {
                /* SET_CDR: TOS=val, SOS=pair → mutate cdr */
                int ptr = (int)sos;
                if (ptr >= 0 && ptr + 1 < HEAP_SIZE) g_heap[ptr + 1] = tos;
                x[S_TOS] = r2; x[S_SOS] = r3; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 2;
                x[S_TYPE_TOS] = x[S_TYPE_R2]; x[S_TYPE_SOS] = x[S_TYPE_R3]; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_PAIR_P) {
                /* PAIR_P: check type tag and heap pointer for compatibility with all paths */
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_PAIR || (tos >= 0 && (int)tos + 1 < g_heap_ptr)) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_NUM_P) {
                /* NUM_P: type_tos == TYPE_NUMBER */
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_NUMBER) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_POPN) {
                /* POPN: pop N values below TOS, keeping TOS */
                int count = (int)(pc >= 0 && pc < n_instr ? prog[pc].operand : 0);
                if (count < 0) count = 0;
                float regs[4] = {x[S_TOS], x[S_SOS], x[S_R2], x[S_R3]};
                float types[4] = {x[S_TYPE_TOS], x[S_TYPE_SOS], x[S_TYPE_R2], x[S_TYPE_R3]};
                /* Remove 'count' items below TOS. TOS stays, SOS = old reg[count+1], etc. */
                for (int i = 1; i < 4; i++) {
                    int src = i + count;
                    if (src < 4) { regs[i] = regs[src]; types[i] = types[src]; }
                    else { regs[i] = 0; types[i] = TYPE_NUMBER; }
                }
                x[S_TOS] = regs[0]; x[S_SOS] = regs[1]; x[S_R2] = regs[2]; x[S_R3] = regs[3];
                x[S_TYPE_TOS] = types[0]; x[S_TYPE_SOS] = types[1]; x[S_TYPE_R2] = types[2]; x[S_TYPE_R3] = types[3];
                x[S_DEPTH] -= count;
            } else if (opcode == OP_TAIL_CALL) {
                /* TAIL_CALL: like CALL but reuses current frame (no frame push) */
                int argc = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (argc < 0) argc = 0;
                if (argc > MEM_SIZE) argc = MEM_SIZE;
                float func_pc = tos;
                float args[4] = {sos, r2, r3, 0};
                for (int i = 0; i < MEM_SIZE; i++) x[S_MEM0+i] = 0;
                for (int i = 0; i < argc && i < MEM_SIZE; i++)
                    x[S_MEM0+i] = args[i];
                x[S_PC] = func_pc;
                x[S_TOS] = 0; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= (1 + argc);
                x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = TYPE_NUMBER;
                x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
            }
            /* ── Closures ── */
            else if (opcode == OP_CLOSURE) {
                /* CLOSURE: operand = func_pc. Allocate closure on heap: [func_pc, n_upvals, upval0, ...] */
                int func_entry = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (g_heap_ptr + 2 <= HEAP_SIZE) {
                    int cptr = g_heap_ptr;
                    g_heap[g_heap_ptr++] = (float)func_entry;  /* entry point */
                    g_heap[g_heap_ptr++] = 0;                  /* n_upvals (set by OPEN_CLOSURE) */
                    /* Push closure ptr, shift down */
                    x[S_TYPE_R3] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_SOS]; x[S_TYPE_SOS] = x[S_TYPE_TOS];
                    x[S_R3] = x[S_R2]; x[S_R2] = x[S_SOS]; x[S_SOS] = x[S_TOS];
                    x[S_TOS] = (float)cptr;
                    x[S_TYPE_TOS] = TYPE_CLOSURE;
                    x[S_DEPTH] += 1;
                }
            } else if (opcode == OP_GET_UPVALUE) {
                int idx = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                float val = 0;
                /* Try closure upvalue array first, fall back to MEM */
                if (g_current_closure_ptr >= 0 && g_current_closure_ptr + 2 + idx < HEAP_SIZE) {
                    val = g_heap[g_current_closure_ptr + 2 + idx];
                } else if (idx >= 0 && idx < MEM_SIZE) {
                    val = x[S_MEM0 + idx];
                }
                /* Push */
                x[S_TYPE_R3] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_SOS]; x[S_TYPE_SOS] = x[S_TYPE_TOS];
                x[S_R3] = x[S_R2]; x[S_R2] = x[S_SOS]; x[S_SOS] = x[S_TOS];
                x[S_TOS] = val;
                x[S_TYPE_TOS] = TYPE_NUMBER;
                x[S_DEPTH] += 1;
            } else if (opcode == OP_SET_UPVALUE) {
                /* SET_UPVALUE: operand = index. Store TOS to upvalue slot, pop. */
                int idx = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (idx >= 0 && idx < MEM_SIZE) x[S_MEM0 + idx] = tos;
                x[S_TOS] = sos; x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_TYPE_TOS] = x[S_TYPE_SOS]; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                x[S_DEPTH] -= 1;
            } else if (opcode == OP_CLOSE_UPVALUE) {
                int idx = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                /* Copy local MEM[idx] to current closure's upvalue slot */
                if (g_current_closure_ptr >= 0 && idx >= 0 && idx < MEM_SIZE &&
                    g_current_closure_ptr + 2 + idx < HEAP_SIZE) {
                    g_heap[g_current_closure_ptr + 2 + idx] = x[S_MEM0 + idx];
                }
            } else if (opcode == OP_OPEN_CLOSURE) {
                /* Set current closure and populate MEM from upvalue array */
                int cptr = (int)x[S_TOS]; /* closure pointer on TOS */
                if (cptr >= 0 && cptr + 1 < HEAP_SIZE) {
                    g_current_closure_ptr = cptr;
                    int n_upvals = (int)g_heap[cptr + 1];
                    for (int i = 0; i < n_upvals && i < MEM_SIZE && cptr + 2 + i < HEAP_SIZE; i++) {
                        x[S_MEM0 + i] = g_heap[cptr + 2 + i];
                    }
                }
            }
            /* ── Vectors ── */
            else if (opcode == OP_VEC_CREATE) {
                /* VEC_CREATE: operand = count. Pop count values, create vector on heap. */
                int count = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (count < 0) count = 0;
                if (count > 4) count = 4; /* max elements from stack registers */
                if (g_heap_ptr + count + 1 <= HEAP_SIZE) {
                    int vptr = g_heap_ptr;
                    g_heap[g_heap_ptr++] = (float)count;  /* length */
                    /* Pop values from stack into vector (TOS is last element) */
                    float vals[4] = {tos, sos, r2, r3};
                    for (int i = 0; i < count && i < 4; i++)
                        g_heap[g_heap_ptr++] = vals[count - 1 - i];
                    /* Adjust stack: pop count values, push vector ptr */
                    x[S_TOS] = (float)vptr;
                    x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                    x[S_DEPTH] -= (count - 1); /* pop count, push 1 */
                    x[S_TYPE_TOS] = TYPE_VECTOR;
                    x[S_TYPE_SOS] = TYPE_NUMBER; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
                }
            } else if (opcode == OP_VEC_REF) {
                /* VEC_REF: TOS=index, SOS=vector_ptr → push vector[index] */
                int vptr = (int)sos;
                int idx = (int)tos;
                if (vptr >= 0 && vptr < HEAP_SIZE) {
                    int len = (int)g_heap[vptr];
                    if (idx >= 0 && idx < len && vptr + 1 + idx < HEAP_SIZE)
                        x[S_TOS] = g_heap[vptr + 1 + idx];
                }
                x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_DEPTH] -= 1;
                x[S_TYPE_TOS] = TYPE_NUMBER; /* element type unknown */
                x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_VEC_SET) {
                /* VEC_SET: TOS=value, SOS=index, R2=vector_ptr → mutate */
                int vptr = (int)r2;
                int idx = (int)sos;
                if (vptr >= 0 && vptr < HEAP_SIZE) {
                    int len = (int)g_heap[vptr];
                    if (idx >= 0 && idx < len && vptr + 1 + idx < HEAP_SIZE)
                        g_heap[vptr + 1 + idx] = tos;
                }
                x[S_TOS] = r3; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                x[S_DEPTH] -= 3;
                x[S_TYPE_TOS] = x[S_TYPE_R3]; x[S_TYPE_SOS] = TYPE_NUMBER; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_VEC_LEN) {
                /* VEC_LEN: TOS=vector_ptr → push length */
                int vptr = (int)tos;
                if (vptr >= 0 && vptr < HEAP_SIZE)
                    x[S_TOS] = g_heap[vptr];
                x[S_TYPE_TOS] = TYPE_NUMBER;
            }
            /* ── Strings (simplified: stored as vectors of char codes) ── */
            else if (opcode == OP_STR_REF) {
                /* STR_REF: TOS=index, SOS=string_ptr → char at index */
                int sptr = (int)sos;
                int idx = (int)tos;
                if (sptr >= 0 && sptr < HEAP_SIZE) {
                    int len = (int)g_heap[sptr];
                    if (idx >= 0 && idx < len && sptr + 1 + idx < HEAP_SIZE)
                        x[S_TOS] = g_heap[sptr + 1 + idx];
                }
                x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                x[S_DEPTH] -= 1;
                x[S_TYPE_TOS] = TYPE_NUMBER; /* char code is a number */
                x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
            } else if (opcode == OP_STR_LEN) {
                /* STR_LEN: TOS=string_ptr → push length */
                int sptr = (int)tos;
                if (sptr >= 0 && sptr < HEAP_SIZE)
                    x[S_TOS] = g_heap[sptr];
                x[S_TYPE_TOS] = TYPE_NUMBER;
            }
            /* ── Type predicates ── */
            else if (opcode == OP_STR_P) {
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_STRING) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_BOOL_P) {
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_BOOL) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_PROC_P) {
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_CLOSURE) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            } else if (opcode == OP_VEC_P) {
                x[S_TOS] = (x[S_TYPE_TOS] == TYPE_VECTOR) ? 1.0f : 0.0f;
                x[S_TYPE_TOS] = TYPE_BOOL;
            }
            /* ── Exceptions ── */
            else if (opcode == OP_PUSH_HANDLER) {
                int handler_pc = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                if (g_exc_count < MAX_EXC_HANDLERS) {
                    g_exc_handlers[g_exc_count].handler_pc = (float)handler_pc;
                    g_exc_handlers[g_exc_count].saved_depth = x[S_DEPTH];
                    for (int i = 0; i < MEM_SIZE; i++)
                        g_exc_handlers[g_exc_count].saved_mem[i] = x[S_MEM0 + i];
                    g_exc_handlers[g_exc_count].saved_tos = x[S_TOS];
                    g_exc_handlers[g_exc_count].saved_sos = x[S_SOS];
                    g_exc_handlers[g_exc_count].saved_r2 = x[S_R2];
                    g_exc_handlers[g_exc_count].saved_r3 = x[S_R3];
                    g_exc_handlers[g_exc_count].saved_type_tos = x[S_TYPE_TOS];
                    g_exc_handlers[g_exc_count].saved_type_sos = x[S_TYPE_SOS];
                    g_exc_handlers[g_exc_count].saved_type_r2 = x[S_TYPE_R2];
                    g_exc_handlers[g_exc_count].saved_type_r3 = x[S_TYPE_R3];
                    g_exc_count++;
                }
            } else if (opcode == OP_POP_HANDLER) {
                if (g_exc_count > 0) g_exc_count--;
            } else if (opcode == OP_GET_EXN) {
                /* Push current exception value */
                x[S_TYPE_R3] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_SOS]; x[S_TYPE_SOS] = x[S_TYPE_TOS];
                x[S_R3] = x[S_R2]; x[S_R2] = x[S_SOS]; x[S_SOS] = x[S_TOS];
                x[S_TOS] = g_current_exn;
                x[S_TYPE_TOS] = TYPE_NUMBER;
                x[S_DEPTH] += 1;
            }
            /* ── Continuations ── */
            else if (opcode == OP_CALLCC) {
                /* CALLCC: capture full continuation state onto the heap and call
                 * the function (TOS) with the continuation as its sole argument.
                 *
                 * Heap layout (16 floats per continuation):
                 *   [ptr+ 0]  return_pc         (resume address)
                 *   [ptr+ 1]  saved_depth        (stack depth at capture)
                 *   [ptr+ 2]  saved_mem[0]
                 *   [ptr+ 3]  saved_mem[1]
                 *   [ptr+ 4]  saved_mem[2]
                 *   [ptr+ 5]  saved_mem[3]
                 *   [ptr+ 6]  saved_tos
                 *   [ptr+ 7]  saved_sos
                 *   [ptr+ 8]  saved_r2
                 *   [ptr+ 9]  saved_r3
                 *   [ptr+10]  saved_type_tags[0] (TOS type)
                 *   [ptr+11]  saved_type_tags[1] (SOS type)
                 *   [ptr+12]  saved_type_tags[2] (R2 type)
                 *   [ptr+13]  saved_type_tags[3] (R3 type)
                 *   [ptr+14]  saved_frame_count
                 *   [ptr+15]  saved_wind_depth
                 */
                #define CONT_SIZE 16
                float func_pc_cc = tos;
                float cont_ptr = -1.0f;
                if (g_heap_ptr + CONT_SIZE <= HEAP_SIZE) {
                    int ptr = g_heap_ptr;
                    g_heap[ptr +  0] = x[S_PC];                /* return PC */
                    g_heap[ptr +  1] = x[S_DEPTH] - 1;         /* depth (pop func ref) */
                    g_heap[ptr +  2] = x[S_MEM0];
                    g_heap[ptr +  3] = x[S_MEM1];
                    g_heap[ptr +  4] = x[S_MEM2];
                    g_heap[ptr +  5] = x[S_MEM3];
                    g_heap[ptr +  6] = sos;                    /* caller TOS (below func) */
                    g_heap[ptr +  7] = r2;                     /* caller SOS */
                    g_heap[ptr +  8] = r3;                     /* caller R2 */
                    g_heap[ptr +  9] = 0;                      /* caller R3 */
                    g_heap[ptr + 10] = x[S_TYPE_SOS];          /* type tags */
                    g_heap[ptr + 11] = x[S_TYPE_R2];
                    g_heap[ptr + 12] = x[S_TYPE_R3];
                    g_heap[ptr + 13] = TYPE_NUMBER;
                    g_heap[ptr + 14] = (float)g_frame_count;
                    g_heap[ptr + 15] = x[S_WIND_DEPTH];
                    g_heap_ptr += CONT_SIZE;
                    cont_ptr = (float)ptr;
                }
                /* Set up callee: continuation object is the sole argument (MEM0) */
                x[S_MEM0] = cont_ptr;
                x[S_MEM1] = 0; x[S_MEM2] = 0; x[S_MEM3] = 0;
                x[S_PC] = func_pc_cc;
                x[S_TOS] = 0; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
                x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = TYPE_NUMBER;
                x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
                x[S_DEPTH] = 0;
            } else if (opcode == OP_INVOKE_CC) {
                /* INVOKE_CC: restore the full continuation state from the heap
                 * and resume execution with TOS as the return value.
                 *
                 * SOS = continuation object (heap pointer), TOS = value to return. */
                float retval_cc = tos;
                float rettype_cc = x[S_TYPE_TOS];
                int cptr = (int)sos;
                if (cptr >= 0 && cptr + CONT_SIZE <= HEAP_SIZE) {
                    /* Restore full machine state from heap */
                    x[S_PC]   = g_heap[cptr +  0];
                    x[S_MEM0] = g_heap[cptr +  2];
                    x[S_MEM1] = g_heap[cptr +  3];
                    x[S_MEM2] = g_heap[cptr +  4];
                    x[S_MEM3] = g_heap[cptr +  5];
                    /* Restore stack: return value on top, then saved registers */
                    x[S_TOS]  = retval_cc;
                    x[S_SOS]  = g_heap[cptr +  6];
                    x[S_R2]   = g_heap[cptr +  7];
                    x[S_R3]   = g_heap[cptr +  8];
                    /* Restore type tags */
                    x[S_TYPE_TOS] = rettype_cc;
                    x[S_TYPE_SOS] = g_heap[cptr + 10];
                    x[S_TYPE_R2]  = g_heap[cptr + 11];
                    x[S_TYPE_R3]  = g_heap[cptr + 12];
                    /* Restore depth (+1 for the return value pushed) */
                    x[S_DEPTH] = g_heap[cptr +  1] + 1;
                    /* Restore frame count and wind depth */
                    g_frame_count = (int)g_heap[cptr + 14];
                    x[S_WIND_DEPTH] = g_heap[cptr + 15];
                    g_wind_depth  = (int)x[S_WIND_DEPTH];
                }
                #undef CONT_SIZE
            }
            /* ── Variadic / dynamic-wind ── */
            else if (opcode == OP_PACK_REST) {
                int n_fixed = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
                /* Cons remaining MEM slots [n_fixed..MEM_SIZE-1] into a list */
                float list_ptr = -1.0f; /* nil */
                for (int i = MEM_SIZE - 1; i >= n_fixed; i--) {
                    if (g_heap_ptr + 2 <= HEAP_SIZE) {
                        int ptr = g_heap_ptr;
                        g_heap[g_heap_ptr++] = x[S_MEM0 + i]; /* car */
                        g_heap[g_heap_ptr++] = list_ptr;        /* cdr */
                        list_ptr = (float)ptr;
                    }
                }
                if (n_fixed >= 0 && n_fixed < MEM_SIZE) {
                    x[S_MEM0 + n_fixed] = list_ptr;
                }
            } else if (opcode == OP_WIND_PUSH) {
                /* TOS = after thunk closure */
                if (g_wind_depth < MAX_WINDS) {
                    g_wind_stack[g_wind_depth].after_thunk_ptr = x[S_TOS];
                    g_wind_stack[g_wind_depth].frame_depth = g_frame_count;
                    g_wind_depth++;
                }
                /* Pop the after thunk */
                x[S_TOS] = x[S_SOS]; x[S_SOS] = x[S_R2]; x[S_R2] = x[S_R3]; x[S_R3] = 0;
                x[S_TYPE_TOS] = x[S_TYPE_SOS]; x[S_TYPE_SOS] = x[S_TYPE_R2]; x[S_TYPE_R2] = x[S_TYPE_R3]; x[S_TYPE_R3] = TYPE_NUMBER;
                x[S_DEPTH] -= 1;
            } else if (opcode == OP_WIND_POP) {
                if (g_wind_depth > 0) g_wind_depth--;
            }
            /* ── Native call dispatch to runtime libraries ── */
            else if (opcode == OP_NATIVE_CALL) {
                int native_id = (pc >= 0 && pc < n_instr) ? prog[pc].operand : -1;
                VmRegionStack* rs = vm_get_regions();

                /* Complex number operations (300-319) */
                if (native_id >= 300 && native_id < 320) {
                    if (native_id == 300) { /* make-rectangular */
                        VmComplex* z = vm_complex_new(rs, sos, tos);
                        if (z && g_heap_ptr + 2 <= HEAP_SIZE) {
                            int ptr = g_heap_ptr;
                            g_heap[g_heap_ptr++] = (float)z->real;
                            g_heap[g_heap_ptr++] = (float)z->imag;
                            /* Pop two args, push result */
                            x[S_TOS] = (float)ptr;
                            x[S_SOS] = r2; x[S_R2] = r3; x[S_R3] = 0;
                            x[S_DEPTH] -= 1;
                            x[S_TYPE_TOS] = (float)VAL_COMPLEX;
                        }
                    } else if (native_id == 302) { /* real-part */
                        int ptr = (int)tos;
                        if (ptr >= 0 && ptr + 1 < HEAP_SIZE) {
                            x[S_TOS] = g_heap[ptr]; /* real part */
                            x[S_TYPE_TOS] = TYPE_NUMBER;
                        }
                    } else if (native_id == 303) { /* imag-part */
                        int ptr = (int)tos;
                        if (ptr >= 0 && ptr + 1 < HEAP_SIZE) {
                            x[S_TOS] = g_heap[ptr + 1]; /* imag part */
                            x[S_TYPE_TOS] = TYPE_NUMBER;
                        }
                    } else if (native_id == 304) { /* magnitude */
                        int ptr = (int)tos;
                        if (ptr >= 0 && ptr + 1 < HEAP_SIZE) {
                            float re = g_heap[ptr], im = g_heap[ptr + 1];
                            x[S_TOS] = sqrtf(re * re + im * im);
                            x[S_TYPE_TOS] = TYPE_NUMBER;
                        }
                    } else if (native_id == 317) { /* complex? */
                        x[S_TOS] = (x[S_TYPE_TOS] == (float)VAL_COMPLEX) ? 1.0f : 0.0f;
                        x[S_TYPE_TOS] = TYPE_BOOL;
                    }
                    /* Other complex ops use the same pattern: read from g_heap, compute, write back */
                }
                /* All other runtime ID ranges: registered but float VM delegates to C exec_loop */
                /* The weight matrix path handles basic opcodes; complex operations dispatch here */
            } else if (opcode >= OP_AD_VAR && opcode <= OP_AD_SQRT) {
                /* AD forward ops: Layer 3 weight matrices handle stack manipulation
                 * and flag setting. The VALUE computation requires tape random-access
                 * which is handled here for correctness. Layer 1 indicator neurons
                 * could compute this in a fully weight-only path (future work). */
                int tlen = (int)roundf(x[S_AD_TAPE_LEN]) - 1; /* already incremented by Layer 4 */
                if (tlen >= 0 && tlen < AD_MAX_TAPE) {
                    float op_type = x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_OP];
                    int li = (int)roundf(x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_LEFT]);
                    int ri = (int)roundf(x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_RIGHT]);
                    float lv = (li >= 0 && li < tlen) ? x[S_AD_TAPE_BASE + li * AD_NODE_FIELDS + AD_F_VALUE] : 0;
                    float rv = (ri >= 0 && ri < tlen) ? x[S_AD_TAPE_BASE + ri * AD_NODE_FIELDS + AD_F_VALUE] : 0;
                    float val = 0, saved = 0;
                    if (fabsf(op_type - AD_OP_VAR) < 0.5f || fabsf(op_type - AD_OP_CONST) < 0.5f) {
                        val = x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_VALUE];
                    } else if (fabsf(op_type - AD_OP_ADD) < 0.5f) { val = lv + rv; }
                    else if (fabsf(op_type - AD_OP_SUB) < 0.5f) { val = lv - rv; }
                    else if (fabsf(op_type - AD_OP_MUL) < 0.5f) { val = lv * rv; }
                    else if (fabsf(op_type - AD_OP_NEG) < 0.5f)     { val = -lv;                          saved = -1.0f; }
                    else if (fabsf(op_type - AD_OP_ABS) < 0.5f)     { val = fabsf(lv);                    saved = (lv > 0) ? 1.0f : (lv < 0) ? -1.0f : 0.0f; }
                    else if (fabsf(op_type - AD_OP_RELU) < 0.5f)    { val = lv > 0 ? lv : 0;             saved = (lv > 0) ? 1.0f : 0.0f; }
                    else if (fabsf(op_type - AD_OP_SIGMOID) < 0.5f) { val = 1.0f/(1.0f+expf(-lv));       saved = val * (1.0f - val); }
                    else if (fabsf(op_type - AD_OP_TANH) < 0.5f)    { val = tanhf(lv);                    saved = 1.0f - val * val; }
                    else if (fabsf(op_type - AD_OP_EXP) < 0.5f)     { val = expf(lv);                     saved = val; }
                    else if (fabsf(op_type - AD_OP_LOG) < 0.5f)     { val = logf(lv > 0 ? lv : 1e-15f);  saved = 1.0f / (fabsf(lv) > 1e-15f ? lv : 1e-15f); }
                    else if (fabsf(op_type - AD_OP_SQRT) < 0.5f)    { val = sqrtf(lv > 0 ? lv : 0);      saved = 1.0f / (2.0f * (fabsf(val) > 1e-15f ? val : 1e-15f)); }
                    x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_VALUE] = val;
                    x[S_AD_TAPE_BASE + tlen * AD_NODE_FIELDS + AD_F_SAVED] = saved;
                }
            } else if (opcode == OP_AD_BACKWARD) {
                /* Seed gradient for backward pass */
                int output_idx = (int)roundf(x[S_AD_CURSOR]);
                int tlen = (int)roundf(x[S_AD_TAPE_LEN]);
                if (output_idx >= 0 && output_idx < tlen)
                    x[S_AD_TAPE_BASE + output_idx * AD_NODE_FIELDS + AD_F_GRAD] = 1.0f;
            } else if (opcode == OP_AD_GRAD) {
                /* Read gradient of tape[TOS] */
                int ni = (int)roundf(x[S_TOS]);
                int tlen = (int)roundf(x[S_AD_TAPE_LEN]);
                float grad = 0;
                if (ni >= 0 && ni < tlen)
                    grad = x[S_AD_TAPE_BASE + ni * AD_NODE_FIELDS + AD_F_GRAD];
                x[S_TOS] = grad;
            }
        }
        x[S_IS_NATIVE] = 0;
    }

    /* IS_CALL: push call frame, jump to function entry
     * Convention: CALL operand = argc
     * Stack before CALL: [func_pc, arg0, arg1, ..., argN-1, ...]
     *   TOS = func_pc (function entry address)
     *   SOS..R3 = args (pushed before the function ref)
     * After CALL: save return PC and caller's memory, set PC to func_pc,
     *   store args in MEM slots, clear stack cache. */
    if (x[S_IS_CALL] > 0.5f) {
        int pc = (int)roundf(x[S_PC]) - 1;
        int argc = (pc >= 0 && pc < n_instr) ? prog[pc].operand : 0;
        if (argc < 0) argc = 0;
        if (argc > MEM_SIZE) argc = MEM_SIZE;
        float func_pc = x[S_TOS];
        float caller_closure = x[S_CUR_CLOSURE];
        /* If func_pc looks like a heap closure pointer, dereference to get entry point */
        int fptr = (int)func_pc;
        if (x[S_TYPE_TOS] == TYPE_CLOSURE &&
            fptr >= 0 && fptr < ARENA_CELLS &&
            ARENA_FIELD(x, fptr, ARENA_F_KIND) == ARENA_KIND_CLOSURE) {
            float candidate = ARENA_FIELD(x, fptr, ARENA_F_CAR_VAL);
            if (candidate >= 0 && candidate < n_instr) func_pc = candidate;
            x[S_CUR_CLOSURE] = (float)fptr;
            g_current_closure_ptr = -1;
        } else if (fptr >= 0 && fptr + 1 < g_heap_ptr && fptr + 1 < HEAP_SIZE && g_heap[fptr] < 10000) {
            /* Check if this is a closure (heap[ptr] = entry_pc, heap[ptr+1] = n_upvals) */
            float candidate = g_heap[fptr];
            if (candidate >= 0 && candidate < n_instr) func_pc = candidate;
            g_current_closure_ptr = fptr; /* track current closure for GET_UPVALUE */
            x[S_CUR_CLOSURE] = -100.0f;
        } else {
            x[S_CUR_CLOSURE] = -100.0f;
        }

        /* Save call frame: return address, memory, and caller's stack below args */
        if (g_frame_count < 64) {
            CallFrame* f = &g_frames[g_frame_count];
            f->return_pc = x[S_PC]; /* already PC+1 */
            f->saved_closure = caller_closure;
            for (int i = 0; i < MEM_SIZE; i++)
                f->saved_mem[i] = x[S_MEM0+i];
            /* Save caller's stack state below the func_pc + args.
             * Stack: [... caller_vals ... arg0 arg1 ... argN func_pc]
             * TOS=func_pc, SOS=last_arg_or_caller, R2=..., R3=...
             * For argc=1: SOS=arg0, R2=caller_val0, R3=caller_val1
             * We save R2/R3 (the caller's values below the args) */
            if (argc == 0) {
                f->saved_tos = x[S_SOS]; f->saved_sos = x[S_R2];
                f->saved_r2 = x[S_R3]; f->saved_r3 = 0;
            } else if (argc == 1) {
                f->saved_tos = x[S_R2]; f->saved_sos = x[S_R3];
                f->saved_r2 = 0; f->saved_r3 = 0;
            } else {
                f->saved_tos = x[S_R3]; f->saved_sos = 0;
                f->saved_r2 = 0; f->saved_r3 = 0;
            }
            f->saved_depth = x[S_DEPTH] - (1 + argc);
            g_frame_count++;
        } else {
            fprintf(stderr, "ERROR: call frame overflow (max 64)\n");
            x[S_HALT] = 1;
        }

        /* Set up callee: args go to MEM0..MEM(argc-1) */
        float args[4] = {x[S_SOS], x[S_R2], x[S_R3], 0};
        for (int i = 0; i < MEM_SIZE; i++) x[S_MEM0+i] = 0;
        for (int i = 0; i < argc && i < MEM_SIZE; i++)
            x[S_MEM0+i] = args[i];

        x[S_PC] = func_pc;
        x[S_TOS] = 0; x[S_SOS] = 0; x[S_R2] = 0; x[S_R3] = 0;
        x[S_TYPE_TOS] = TYPE_NUMBER; x[S_TYPE_SOS] = TYPE_NUMBER;
        x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
        x[S_DEPTH] = 0;
        x[S_IS_CALL] = 0;
    }

    /* IS_RET: pop call frame, restore caller state, push return value
     * TOS = return value */
    if (x[S_IS_RET] > 0.5f) {
        float retval = x[S_TOS];
        float rettype = x[S_TYPE_TOS];

        if (g_frame_count > 0) {
            g_frame_count--;
            CallFrame* f = &g_frames[g_frame_count];
            x[S_PC] = f->return_pc;
            for (int i = 0; i < MEM_SIZE; i++)
                x[S_MEM0+i] = f->saved_mem[i];
            /* Restore caller's stack with return value pushed on top */
            x[S_TOS] = retval;
            x[S_TYPE_TOS] = rettype;
            x[S_SOS] = f->saved_tos;
            x[S_R2] = f->saved_sos;
            x[S_R3] = f->saved_r2;
            x[S_TYPE_SOS] = TYPE_NUMBER; x[S_TYPE_R2] = TYPE_NUMBER; x[S_TYPE_R3] = TYPE_NUMBER;
            x[S_DEPTH] = f->saved_depth + 1; /* +1 for return value */
            x[S_CUR_CLOSURE] = f->saved_closure;
        }
        x[S_IS_RET] = 0;
    }
}

/**
 * @brief Run a program to completion (or a step cap) on the simulated
 *        transformer (layer0_attention() through layer5_writeback(), plus
 *        exec_loop_postprocess()), verifying the same behaviour as
 *        run_reference() but via the gated-FFN weight-computation form.
 *        Collects up to @p max_out output values and, if g_trace_sim_fp is
 *        set, emits a JSONL trace line per step.
 * @return The number of outputs produced.
 */
