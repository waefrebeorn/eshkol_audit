static void execute_chunk(FuncChunk* chunk) {
    /* Allocate VM on heap (too large for stack) */
    Value* stack = (Value*)calloc(STACK_SIZE, sizeof(Value));
    HeapObject* heap = (HeapObject*)calloc(HEAP_SIZE, sizeof(HeapObject));
    CallFrame* frames = (CallFrame*)calloc(MAX_FRAMES, sizeof(CallFrame));
    if (!stack || !heap || !frames) {
        fprintf(stderr, "ERROR: VM allocation failed\n");
        free(stack); free(heap); free(frames);
        return;
    }
    int32_t sp = 0, pc = 0, fp = 0, heap_next = 0;
    int frame_count = 0, halted = 0, error = 0;

    #define PUSH(v) do { if(sp>=STACK_SIZE){fprintf(stderr,"STACK OVERFLOW at PC=%d\n",pc);error=1;}else stack[sp++]=(v); } while(0)
    #define POP() (sp > 0 ? stack[--sp] : (fprintf(stderr,"STACK UNDERFLOW at PC=%d\n",pc),error=1, (Value){.type=VAL_NIL}))
    #define PEEK(off) ((sp-1-(off)) >= 0 ? stack[sp-1-(off)] : (Value){.type=VAL_NIL})
    #define AS_NUM(v) ((v).type==VAL_INT?(double)(v).as.i:(v).as.f)
    #define NUM_VAL(r) ((r)==(int64_t)(r)&&fabs(r)<1e15 ? INT_VAL((int64_t)(r)) : FLOAT_VAL(r))
    /* R7RS: only #f is falsy. Empty list '() is truthy. */
    #define IS_FALSY(v) ((v).type==VAL_BOOL && !(v).as.b)
    #define HALLOC() (heap_next < HEAP_SIZE ? heap_next++ : (printf("HEAP OOM\n"),error=1,-1))

    /* Exception handler stack (for guard/raise) */
    #define MAX_HANDLERS 32
    struct { int32_t handler_pc; int32_t saved_sp; int32_t saved_fp;
             int32_t saved_frame_count; } exc_handlers[MAX_HANDLERS];
    int handler_count = 0;
    Value current_exn = {.type = VAL_NIL}; /* current exception value (set by raise) */

    /* Dynamic-wind stack: tracks active before/after thunks */
    #define MAX_WINDS 32
    struct { Value after; /* after thunk closure */ } wind_stack[MAX_WINDS];
    int wind_depth = 0;

    /* Pending continuation invocation (for wind unwinding) */
    int pending_cc = -1;       /* heap index of pending continuation, -1 = none */
    Value pending_cc_result = {.type = VAL_NIL};

    int64_t insn_count = 0;
    int max_depth = 0;
    int trace_on = g_trace_on; /* set by --trace flag */
    while (!halted && !error && pc < chunk->code_len) {
        if (frame_count > max_depth) max_depth = frame_count;
        { int64_t max_insn = 10000000LL;
          const char* env_max = getenv("ESHKOL_VM_MAX_INSN");
          if (env_max) max_insn = atoll(env_max);
          if (++insn_count > max_insn) { printf("RUNAWAY (%lld insns, depth=%d, heap=%d)\n", (long long)max_insn, max_depth, heap_next); error=1; break; }
        }
        if (trace_on && insn_count < 500) {
            printf("  [%04d] op=%2d sp=%d fp=%d", pc-1, chunk->code[pc-1].op, sp, fp);
            if (sp > 0) { Value t = stack[sp-1]; printf(" TOS="); if(t.type==VAL_INT)printf("%lld",(long long)t.as.i); else if(t.type==VAL_VECTOR)printf("#vec@%d",t.as.ptr); else if(t.type==VAL_CLOSURE)printf("<cl@%d>",t.as.ptr); else if(t.type==VAL_BOOL)printf(t.as.b?"#t":"#f"); else if(t.type==VAL_NIL)printf("nil"); else printf("?"); }
            printf("\n");
        }
        Instr ins = chunk->code[pc++];
        switch (ins.op) {
        case OP_NOP: break;
        case OP_CONST: PUSH(chunk->constants[ins.operand]); break;
        case OP_NIL: PUSH(((Value){.type=VAL_NIL})); break;
        case OP_TRUE: PUSH(((Value){.type=VAL_BOOL,.as.b=1})); break;
        case OP_FALSE: PUSH(((Value){.type=VAL_BOOL,.as.b=0})); break;
        case OP_POP: sp--; break;
        case OP_DUP: { Value v = PEEK(0); PUSH(v); break; }

        case OP_ADD: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)+AS_NUM(b))); break; }
        case OP_SUB: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)-AS_NUM(b))); break; }
        case OP_MUL: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)*AS_NUM(b))); break; }
        case OP_DIV: {
            Value b=POP(),a=POP(); double bv=AS_NUM(b);
            if (bv == 0.0 && b.type == VAL_INT) {
                /* Integer division by zero: raise error */
                Value exn_msg = INT_VAL(0); /* "division by zero" */
                if (handler_count > 0) {
                    handler_count--;
                    sp = exc_handlers[handler_count].saved_sp;
                    fp = exc_handlers[handler_count].saved_fp;
                    frame_count = exc_handlers[handler_count].saved_frame_count;
                    pc = exc_handlers[handler_count].handler_pc;
                    current_exn = exn_msg;
                } else { printf("ERROR: division by zero\n"); error=1; }
            } else {
                PUSH(NUM_VAL(AS_NUM(a)/bv));
            }
            break;
        }
        case OP_MOD: {
            Value b=POP(),a=POP();
            double bv=AS_NUM(b);
            if (bv == 0.0 && b.type == VAL_INT) {
                if (handler_count > 0) {
                    handler_count--;
                    sp = exc_handlers[handler_count].saved_sp;
                    fp = exc_handlers[handler_count].saved_fp;
                    frame_count = exc_handlers[handler_count].saved_frame_count;
                    pc = exc_handlers[handler_count].handler_pc;
                    current_exn = INT_VAL(0);
                } else { printf("ERROR: modulo by zero\n"); error=1; }
            } else {
                double r = fmod(AS_NUM(a), bv);
                if (r != 0 && ((r > 0) != (bv > 0))) r += bv;
                PUSH(NUM_VAL(r));
            }
            break;
        }
        case OP_NEG: { Value a=POP(); PUSH(NUM_VAL(-AS_NUM(a))); break; }
        case OP_ABS: { Value a=POP(); PUSH(NUM_VAL(fabs(AS_NUM(a)))); break; }

        case OP_EQ: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)==AS_NUM(b))})); break; }
        case OP_LT: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)<AS_NUM(b))})); break; }
        case OP_GT: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)>AS_NUM(b))})); break; }
        case OP_LE: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)<=AS_NUM(b))})); break; }
        case OP_GE: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)>=AS_NUM(b))})); break; }
        case OP_NOT: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=IS_FALSY(a)})); break; }

        case OP_GET_LOCAL: {
            int idx = fp + ins.operand;
            if (idx < 0 || idx >= sp) { printf("GET_LOCAL out of bounds: fp=%d op=%d sp=%d\n", fp, ins.operand, sp); error=1; break; }
            PUSH(stack[idx]); break;
        }
        case OP_SET_LOCAL: {
            int idx = fp + ins.operand;
            if (idx < 0 || idx >= STACK_SIZE) { printf("SET_LOCAL out of bounds\n"); error=1; break; }
            stack[idx] = PEEK(0); sp--; break;
        }
        case OP_GET_UPVALUE: {
            Value cl = stack[fp - 1];
            if (cl.type == VAL_CLOSURE) {
                int uv = ins.operand;
                if (uv < 0 || uv >= 16) { printf("UPVALUE index out of bounds: %d\n", uv); error=1; break; }
                int32_t open_slot = heap[cl.as.ptr].closure.open_slots[uv];
                if (open_slot >= 0) {
                    /* Open upvalue: read from stack slot (sees latest value) */
                    PUSH(stack[open_slot]);
                } else {
                    /* Closed upvalue: read from captured value */
                    PUSH(heap[cl.as.ptr].closure.upvalues[uv]);
                }
            } else PUSH(((Value){.type=VAL_NIL}));
            break;
        }
        case OP_SET_UPVALUE: {
            Value cl = stack[fp - 1];
            if (cl.type == VAL_CLOSURE) {
                int uv = ins.operand;
                if (uv < 0 || uv >= 16) { printf("UPVALUE index out of bounds: %d\n", uv); error=1; break; }
                int32_t open_slot = heap[cl.as.ptr].closure.open_slots[uv];
                if (open_slot >= 0) {
                    stack[open_slot] = PEEK(0);  /* write through to stack */
                } else {
                    heap[cl.as.ptr].closure.upvalues[uv] = PEEK(0);
                }
            }
            sp--; break;
        }

        case OP_CLOSURE: {
            int ci = ins.operand & 0xFFFF;
            int nu = (ins.operand >> 16) & 0xFF;
            if (nu > 16) nu = 16;
            int32_t func_pc = (int32_t)chunk->constants[ci].as.i;
            int32_t ptr = HALLOC();
            if (ptr < 0) break;
            heap[ptr].type = HEAP_CLOSURE;
            heap[ptr].closure.func_pc = func_pc;
            heap[ptr].closure.n_upvalues = nu;
            for (int i = nu - 1; i >= 0; i--) {
                heap[ptr].closure.upvalues[i] = POP();
                heap[ptr].closure.open_slots[i] = -1; /* closed by default */
            }
            PUSH(((Value){.type=VAL_CLOSURE,.as.ptr=ptr}));
            break;
        }

        case OP_CALL: {
            int argc = ins.operand;
            Value func = stack[sp - 1 - argc];
            /* Handle continuation invocation */
            if (func.type == VAL_CONTINUATION) {
                Value result = (argc >= 1) ? stack[sp - 1] : ((Value){.type=VAL_NIL});
                int32_t cc_ptr = func.as.ptr;
                if (heap[cc_ptr].continuation.used) {
                    printf("ERROR: continuation invoked more than once (single-shot)\n");
                    error = 1; break;
                }
                heap[cc_ptr].continuation.used = 1;
                /* Check if we need to unwind dynamic-wind frames */
                int target_wind = heap[cc_ptr].continuation.saved_wind_depth;
                if (wind_depth > target_wind) {
                    /* Need to unwind: save continuation for later, call first after thunk */
                    pending_cc = cc_ptr;
                    pending_cc_result = result;
                    wind_depth--;
                    Value after = wind_stack[wind_depth].after;
                    if (after.type == VAL_CLOSURE) {
                        /* Call after thunk — when it returns, OP_RETURN will check pending_cc */
                        PUSH(after);
                        if (frame_count >= MAX_FRAMES) { error=1; break; }
                        frames[frame_count].return_pc = pc;
                        frames[frame_count].return_fp = fp;
                        frames[frame_count].heap_mark = heap_next;
                        frames[frame_count].force_promise_ptr = -1;
                        frame_count++;
                        fp = sp;
                        pc = heap[after.as.ptr].closure.func_pc;
                    }
                    break;
                }
                /* No wind unwinding needed — restore directly */
                int save_n = heap[cc_ptr].continuation.saved_sp;
                if (save_n > 256) save_n = 256;
                memcpy(stack, heap[cc_ptr].continuation.saved_stack, save_n * sizeof(Value));
                pc = heap[cc_ptr].continuation.saved_pc;
                sp = heap[cc_ptr].continuation.saved_sp;
                fp = heap[cc_ptr].continuation.saved_fp;
                frame_count = heap[cc_ptr].continuation.saved_frame_count;
                PUSH(result);
                break;
            }
            if (func.type != VAL_CLOSURE) { printf("CALL non-function\n"); error=1; break; }
            if (frame_count >= MAX_FRAMES) { printf("FRAME OVERFLOW\n"); error=1; break; }
            frames[frame_count].return_pc = pc;
            frames[frame_count].return_fp = fp;
            frames[frame_count].heap_mark = heap_next;  /* OALR: save region boundary */
            frames[frame_count].force_promise_ptr = -1;
            frame_count++;
            fp = sp - argc;
            pc = heap[func.as.ptr].closure.func_pc;
            break;
        }

        case OP_TAIL_CALL: {
            int argc = ins.operand;
            Value func = stack[sp - 1 - argc];
            /* Handle continuation invocation (same as OP_CALL path) */
            if (func.type == VAL_CONTINUATION) {
                Value result = (argc >= 1) ? stack[sp - 1] : ((Value){.type=VAL_NIL});
                int32_t cc_ptr = func.as.ptr;
                if (heap[cc_ptr].continuation.used) {
                    printf("ERROR: continuation invoked more than once (single-shot)\n");
                    error = 1; break;
                }
                heap[cc_ptr].continuation.used = 1;
                int target_wind = heap[cc_ptr].continuation.saved_wind_depth;
                if (wind_depth > target_wind) {
                    pending_cc = cc_ptr;
                    pending_cc_result = result;
                    wind_depth--;
                    Value after = wind_stack[wind_depth].after;
                    if (after.type == VAL_CLOSURE) {
                        PUSH(after);
                        if (frame_count >= MAX_FRAMES) { error=1; break; }
                        frames[frame_count].return_pc = pc;
                        frames[frame_count].return_fp = fp;
                        frames[frame_count].heap_mark = heap_next;
                        frames[frame_count].force_promise_ptr = -1;
                        frame_count++;
                        fp = sp;
                        pc = heap[after.as.ptr].closure.func_pc;
                    }
                    break;
                }
                int save_n = heap[cc_ptr].continuation.saved_sp;
                if (save_n > 256) save_n = 256;
                memcpy(stack, heap[cc_ptr].continuation.saved_stack, save_n * sizeof(Value));
                pc = heap[cc_ptr].continuation.saved_pc;
                sp = heap[cc_ptr].continuation.saved_sp;
                fp = heap[cc_ptr].continuation.saved_fp;
                frame_count = heap[cc_ptr].continuation.saved_frame_count;
                PUSH(result);
                break;
            }
            if (func.type != VAL_CLOSURE) { error=1; break; }
            /* Update the closure reference at fp-1 so GET_UPVALUE works
             * correctly for the callee (not the caller's closure) */
            stack[fp - 1] = func;
            for (int i = 0; i < argc; i++)
                stack[fp + i] = stack[sp - argc + i];
            sp = fp + argc;
            pc = heap[func.as.ptr].closure.func_pc;
            break;
        }

        case OP_RETURN: {
            Value result = POP();
            if (frame_count <= 0) { PUSH(result); halted = 1; break; }
            frame_count--;
            int32_t mark = frames[frame_count].heap_mark;

            /* OALR region cleanup disabled — it's unsafe when side-effecting
             * functions store heap values in persistent structures (hash tables,
             * vectors, set! targets). Full OALR requires escape analysis.
             * The heap is large enough (4M objects) that this isn't a problem. */
            (void)mark;

            /* Check if returning from a force — memoize the result */
            if (frames[frame_count].force_promise_ptr >= 0) {
                int32_t pp = frames[frame_count].force_promise_ptr;
                if (pp < HEAP_SIZE && heap[pp].type == HEAP_VECTOR) {
                    heap[pp].vector.items[0] = (Value){.type=VAL_BOOL,.as.b=1}; /* mark forced */
                    heap[pp].vector.items[1] = result; /* cache the value */
                }
                frames[frame_count].force_promise_ptr = -1;
            }

            sp = fp - 1;
            fp = frames[frame_count].return_fp;
            pc = frames[frame_count].return_pc;
            PUSH(result);

            /* Check for pending continuation (wind unwinding in progress) */
            if (pending_cc >= 0) {
                int target_wind = heap[pending_cc].continuation.saved_wind_depth;
                if (wind_depth > target_wind) {
                    /* More after thunks to call */
                    POP(); /* discard after thunk's return value */
                    wind_depth--;
                    Value after = wind_stack[wind_depth].after;
                    if (after.type == VAL_CLOSURE) {
                        PUSH(after);
                        if (frame_count >= MAX_FRAMES) { error=1; break; }
                        frames[frame_count].return_pc = pc;
                        frames[frame_count].return_fp = fp;
                        frames[frame_count].heap_mark = heap_next;
                        frames[frame_count].force_promise_ptr = -1;
                        frame_count++;
                        fp = sp;
                        pc = heap[after.as.ptr].closure.func_pc;
                    }
                } else {
                    /* All after thunks done — invoke the pending continuation.
                     * Do NOT memcpy — preserve after thunk mutations.
                     * Only restore sp/fp/frame_count/pc. */
                    POP(); /* discard last after thunk's return value */
                    int32_t cc_ptr = pending_cc;
                    Value cc_result = pending_cc_result;
                    pending_cc = -1;
                    pc = heap[cc_ptr].continuation.saved_pc;
                    sp = heap[cc_ptr].continuation.saved_sp;
                    fp = heap[cc_ptr].continuation.saved_fp;
                    frame_count = heap[cc_ptr].continuation.saved_frame_count;
                    wind_depth = heap[cc_ptr].continuation.saved_wind_depth;
                    PUSH(cc_result);
                }
            }
            break;
        }

        case OP_JUMP:
            if (ins.operand < 0 || ins.operand >= chunk->code_len) { printf("JUMP target out of bounds: %d\n", ins.operand); error=1; break; }
            pc = ins.operand; break;
        case OP_JUMP_IF_FALSE: {
            Value v=POP();
            if(IS_FALSY(v)) {
                if (ins.operand < 0 || ins.operand >= chunk->code_len) { printf("JUMP_IF_FALSE target out of bounds: %d\n", ins.operand); error=1; break; }
                pc=ins.operand;
            }
            break;
        }
        case OP_LOOP:
            if (ins.operand < 0 || ins.operand >= chunk->code_len) { printf("LOOP target out of bounds: %d\n", ins.operand); error=1; break; }
            pc = ins.operand; break;

        case OP_CONS: {
            Value car=POP(), cdr=POP(); /* TOS=car, SOS=cdr */
            int32_t ptr = HALLOC(); if(ptr<0) break;
            heap[ptr].type = HEAP_CONS;
            heap[ptr].cons.car = car;
            heap[ptr].cons.cdr = cdr;
            PUSH(((Value){.type=VAL_PAIR,.as.ptr=ptr}));
            break;
        }
        case OP_CAR: { Value p=POP(); if(p.type!=VAL_PAIR){error=1;break;} PUSH(heap[p.as.ptr].cons.car); break; }
        case OP_CDR: { Value p=POP(); if(p.type!=VAL_PAIR){error=1;break;} PUSH(heap[p.as.ptr].cons.cdr); break; }
        case OP_NULL_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_NIL)})); break; }

        case OP_PRINT: {
            Value v = POP();
            printf("  → ");
            print_value(v, heap, 0, 1);
            printf("\n");
            break;
        }

        case OP_CLOSE_UPVALUE: {
            /* Patch closure's upvalue[operand] = the closure itself (self-reference) */
            Value cl = PEEK(0);
            if (cl.type == VAL_CLOSURE) {
                heap[cl.as.ptr].closure.upvalues[ins.operand] = cl;
            }
            break;
        }

        /* Vectors */
        case OP_VEC_CREATE: {
            int count = ins.operand;
            int32_t ptr = HALLOC(); if (ptr < 0) break;
            heap[ptr].type = HEAP_VECTOR;
            /* P0: vector.items[] is a fixed 64-slot inline array. `count` comes
               straight from the instruction operand, so a vector/struct literal
               with >64 elements would write past items[] into the next heap
               object. Clamp the stored count to capacity; still POP the full
               count so the operand stack stays balanced. */
            const int cap = (int)(sizeof(heap[ptr].vector.items) /
                                  sizeof(heap[ptr].vector.items[0]));
            int stored = count > cap ? cap : count;
            heap[ptr].vector.len = stored;
            for (int i = count - 1; i >= 0; i--) {
                Value v = POP();
                if (i < stored) heap[ptr].vector.items[i] = v;
            }
            PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
            break;
        }
        case OP_VEC_REF: {
            Value idx_v = POP(), vec_v = POP();
            if (vec_v.type != VAL_VECTOR) { error=1; break; }
            int idx = (int)AS_NUM(idx_v);
            if (idx < 0 || idx >= heap[vec_v.as.ptr].vector.len) { printf("VEC_REF out of bounds\n"); error=1; break; }
            PUSH(heap[vec_v.as.ptr].vector.items[idx]);
            break;
        }
        case OP_VEC_SET: {
            Value val = POP(), idx_v = POP(), vec_v = POP();
            if (vec_v.type != VAL_VECTOR) { error=1; break; }
            int idx = (int)AS_NUM(idx_v);
            if (idx >= 0 && idx < heap[vec_v.as.ptr].vector.len)
                heap[vec_v.as.ptr].vector.items[idx] = val;
            break;
        }
        case OP_VEC_LEN: {
            Value v = POP();
            if (v.type == VAL_VECTOR) PUSH(INT_VAL(heap[v.as.ptr].vector.len));
            else PUSH(INT_VAL(0));
            break;
        }

        /* Strings */
        case OP_STR_REF: {
            Value idx_v = POP(), str_v = POP();
            if (str_v.type != VAL_STRING) { error=1; break; }
            int idx = (int)AS_NUM(idx_v);
            if (idx >= 0 && idx < heap[str_v.as.ptr].string.len)
                PUSH(INT_VAL(heap[str_v.as.ptr].string.data[idx]));
            else PUSH(INT_VAL(0));
            break;
        }
        case OP_STR_LEN: {
            Value v = POP();
            if (v.type == VAL_STRING) PUSH(INT_VAL(heap[v.as.ptr].string.len));
            else PUSH(INT_VAL(0));
            break;
        }

        /* Type predicates */
        case OP_PAIR_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_PAIR)})); break; }
        case OP_NUM_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_INT||v.type==VAL_FLOAT)})); break; }
        case OP_STR_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_STRING)})); break; }
        case OP_BOOL_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_BOOL)})); break; }
        case OP_PROC_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_CLOSURE)})); break; }
        case OP_VEC_P: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_VECTOR)})); break; }

        /* Set mutations */
        case OP_SET_CAR: { Value val=POP(), p=POP(); if(p.type==VAL_PAIR) heap[p.as.ptr].cons.car=val; break; }
        case OP_SET_CDR: { Value val=POP(), p=POP(); if(p.type==VAL_PAIR) heap[p.as.ptr].cons.cdr=val; break; }

        /* call/cc: capture current continuation, call TOS with it */
        case OP_CALLCC: {
            Value proc = POP(); /* the procedure to call with the continuation */
            if (proc.type != VAL_CLOSURE) { printf("CALLCC non-procedure\n"); error=1; break; }
            /* Capture continuation: save pc, sp, fp, frame_count, and stack snapshot */
            int32_t cc_ptr = HALLOC(); if (cc_ptr < 0) break;
            heap[cc_ptr].type = HEAP_CONTINUATION;
            heap[cc_ptr].continuation.saved_pc = pc;
            heap[cc_ptr].continuation.saved_sp = sp;
            heap[cc_ptr].continuation.saved_fp = fp;
            heap[cc_ptr].continuation.saved_frame_count = frame_count;
            heap[cc_ptr].continuation.used = 0;
            heap[cc_ptr].continuation.saved_wind_depth = wind_depth;
            int save_n = sp < 256 ? sp : 256;
            memcpy(heap[cc_ptr].continuation.saved_stack, stack, save_n * sizeof(Value));
            /* Create a closure that invokes the continuation when called */
            Value cc_val = {.type = VAL_CONTINUATION, .as.ptr = cc_ptr};
            /* Call proc with the continuation as argument */
            PUSH(proc); /* function */
            PUSH(cc_val); /* argument = continuation */
            /* Inline CALL 1 */
            if (frame_count >= MAX_FRAMES) { error=1; break; }
            frames[frame_count].return_pc = pc;
            frames[frame_count].return_fp = fp;
            frames[frame_count].heap_mark = heap_next;
            frames[frame_count].force_promise_ptr = -1;
            frame_count++;
            fp = sp - 1;
            pc = heap[proc.as.ptr].closure.func_pc;
            break;
        }

        /* Invoke a continuation with a value */
        case OP_INVOKE_CC: {
            /* Not needed as opcode — continuations are invoked via CALL */
            break;
        }

        /* Exception handling: push handler (save unwind point) */
        case OP_PUSH_HANDLER: {
            if (handler_count >= MAX_HANDLERS) { printf("TOO MANY HANDLERS\n"); error=1; break; }
            exc_handlers[handler_count].handler_pc = ins.operand;
            exc_handlers[handler_count].saved_sp = sp;
            exc_handlers[handler_count].saved_fp = fp;
            exc_handlers[handler_count].saved_frame_count = frame_count;
            handler_count++;
            break;
        }

        /* Exception handling: pop handler (normal guard exit) */
        case OP_POP_HANDLER: {
            if (handler_count > 0) handler_count--;
            break;
        }

        /* Push current exception value (set by most recent raise) */
        case OP_GET_EXN: {
            PUSH(current_exn);
            break;
        }

        /* Wind stack push: store after thunk for dynamic-wind unwinding */
        case OP_WIND_PUSH: {
            Value after = POP();
            if (wind_depth < MAX_WINDS) {
                wind_stack[wind_depth].after = after;
                wind_depth++;
            }
            break;
        }

        /* Wind stack pop */
        case OP_WIND_POP: {
            if (wind_depth > 0) wind_depth--;
            break;
        }

        /* Pack rest arguments into a list.
         * operand = n_fixed params. Args from fp+n_fixed to sp-1 become a list.
         * After packing: sp = fp + n_fixed + 1, stack[fp+n_fixed] = rest list. */
        case OP_PACK_REST: {
            int n_fixed = ins.operand;
            int rest_start = fp + n_fixed;
            int rest_count = sp - rest_start;
            /* Build list from the rest args (right to left) */
            Value rest_list = {.type = VAL_NIL};
            for (int ri = sp - 1; ri >= rest_start; ri--) {
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_CONS;
                heap[ptr].cons.car = stack[ri];
                heap[ptr].cons.cdr = rest_list;
                rest_list = (Value){.type = VAL_PAIR, .as.ptr = ptr};
            }
            /* Store rest list at fp+n_fixed, set sp */
            stack[rest_start] = rest_list;
            sp = rest_start + 1;
            break;
        }

        /* Open closure: like CLOSURE but upvalues are stack slot references (not captured values).
         * The stack has [slot_idx_0, slot_idx_1, ...] pushed as INT values.
         * Each slot index tells GET_UPVALUE where to read from the ENCLOSING stack. */
        case OP_OPEN_CLOSURE: {
            int ci = ins.operand & 0xFFFF;
            int nu = (ins.operand >> 16) & 0xFF;
            int32_t func_pc = (int32_t)chunk->constants[ci].as.i;
            int32_t ptr = HALLOC();
            if (ptr < 0) break;
            heap[ptr].type = HEAP_CLOSURE;
            heap[ptr].closure.func_pc = func_pc;
            heap[ptr].closure.n_upvalues = nu;
            for (int i = nu - 1; i >= 0; i--) {
                Value slot_val = POP();
                int32_t slot_idx = (int32_t)slot_val.as.i;
                heap[ptr].closure.upvalues[i] = ((Value){.type=VAL_NIL}); /* placeholder */
                heap[ptr].closure.open_slots[i] = slot_idx; /* open reference to stack slot */
            }
            PUSH(((Value){.type=VAL_CLOSURE,.as.ptr=ptr}));
            break;
        }

        /* Scope cleanup: pop N values below TOS, keeping TOS */
        case OP_POPN: {
            int n = ins.operand;
            if (n > 0 && sp > n) {
                Value result = stack[sp - 1]; /* save TOS */
                sp -= n;                       /* remove N values */
                stack[sp - 1] = result;        /* put result back at new TOS */
            }
            break;
        }

        /* Native call registry */
        case OP_NATIVE_CALL: {
            int fid = ins.operand;
            switch (fid) {
            /* Math functions (single arg on stack) */
            case 20: { Value a=POP(); PUSH(FLOAT_VAL(sin(AS_NUM(a)))); break; }   /* sin */
            case 21: { Value a=POP(); PUSH(FLOAT_VAL(cos(AS_NUM(a)))); break; }   /* cos */
            case 22: { Value a=POP(); PUSH(FLOAT_VAL(tan(AS_NUM(a)))); break; }   /* tan */
            case 23: { Value a=POP(); PUSH(FLOAT_VAL(exp(AS_NUM(a)))); break; }   /* exp */
            case 24: { Value a=POP(); PUSH(FLOAT_VAL(log(AS_NUM(a)))); break; }   /* log */
            case 25: { Value a=POP(); PUSH(FLOAT_VAL(sqrt(AS_NUM(a)))); break; }  /* sqrt */
            case 26: { Value a=POP(); PUSH(NUM_VAL(floor(AS_NUM(a)))); break; }   /* floor */
            case 27: { Value a=POP(); PUSH(NUM_VAL(ceil(AS_NUM(a)))); break; }    /* ceiling */
            case 28: { Value a=POP(); PUSH(NUM_VAL(round(AS_NUM(a)))); break; }   /* round */
            case 29: { Value a=POP(); PUSH(FLOAT_VAL(asin(AS_NUM(a)))); break; }  /* asin */
            case 30: { Value a=POP(); PUSH(FLOAT_VAL(acos(AS_NUM(a)))); break; }  /* acos */
            case 31: { Value a=POP(); PUSH(FLOAT_VAL(atan(AS_NUM(a)))); break; }  /* atan */
            /* Two-arg math */
            case 32: { Value b=POP(),a=POP(); PUSH(FLOAT_VAL(pow(AS_NUM(a),AS_NUM(b)))); break; } /* expt */
            case 33: { Value b=POP(),a=POP(); PUSH(NUM_VAL(fmin(AS_NUM(a),AS_NUM(b)))); break; }  /* min */
            case 34: { Value b=POP(),a=POP(); PUSH(NUM_VAL(fmax(AS_NUM(a),AS_NUM(b)))); break; }  /* max */
            case 35: { Value a=POP(); PUSH(NUM_VAL(fabs(AS_NUM(a)))); break; }  /* abs */
            case 36: { Value b=POP(),a=POP(); PUSH(NUM_VAL(fmod(AS_NUM(a),AS_NUM(b)))); break; }  /* modulo */
            case 37: { Value b=POP(),a=POP(); PUSH(NUM_VAL(fmod(AS_NUM(a),AS_NUM(b)))); break; }  /* remainder */
            case 38: { Value b=POP(),a=POP(); PUSH(NUM_VAL(floor(AS_NUM(a)/AS_NUM(b)))); break; } /* quotient */
            /* Numeric predicates */
            case 40: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)>0)})); break; }  /* positive? */
            case 41: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)<0)})); break; }  /* negative? */
            case 42: { Value a=POP(); double v=AS_NUM(a); PUSH(((Value){.type=VAL_BOOL,.as.b=((int64_t)v%2!=0)})); break; } /* odd? */
            case 43: { Value a=POP(); double v=AS_NUM(a); PUSH(((Value){.type=VAL_BOOL,.as.b=((int64_t)v%2==0)})); break; } /* even? */
            case 44: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(AS_NUM(a)==0)})); break; }  /* zero? */
            case 45: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_NIL)})); break; } /* null? */
            case 46: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_PAIR)})); break; } /* pair? */
            case 47: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_INT||a.type==VAL_FLOAT)})); break; } /* number? */
            case 48: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_BOOL)})); break; } /* boolean? */
            case 49: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_CLOSURE)})); break; } /* procedure? */
            case 50: { Value a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.type==VAL_VECTOR)})); break; } /* vector? */
            /* String operations */
            /* native 50 is vector? (defined above) */
            /* string-length is now native 56 */
            case 51: { /* number->string */
                Value v=POP();
                int32_t ptr = HALLOC(); if(ptr<0) break;
                heap[ptr].type = HEAP_STRING;
                if (v.type==VAL_INT) heap[ptr].string.len = snprintf(heap[ptr].string.data, 255, "%lld", (long long)v.as.i);
                else heap[ptr].string.len = snprintf(heap[ptr].string.data, 255, "%.6g", v.as.f);
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr}));
                break;
            }
            /* Display with newline */
            case 53: { /* build-string: stack has [..., len, c0, c1, ..., cN-1]
                       * len was pushed first, then N char values (as numbers).
                       * TOS = cN-1, below that c0..cN-2, below those = len.
                       * We peek at len to know how many chars to pop. */
                /* Scan for the length value below the chars on the stack.
                 * len is at position sp - N - 1 where N = len value itself. */
                int slen = 0;
                for (int try_len = 0; try_len < 256; try_len++) {
                    int len_pos = sp - try_len - 1;
                    if (len_pos >= 0 && stack[len_pos].type == VAL_INT && stack[len_pos].as.i == try_len) {
                        slen = try_len;
                        break;
                    }
                }
                if (slen > 255) slen = 255;
                /* Pop chars in reverse (TOS is last char) */
                char buf[256];
                for (int i = slen - 1; i >= 0; i--) {
                    Value ch = POP();
                    buf[i] = (char)(int)AS_NUM(ch);
                }
                POP(); /* pop len */
                buf[slen] = 0;
                int32_t sptr = HALLOC(); if (sptr < 0) break;
                heap[sptr].type = HEAP_STRING;
                heap[sptr].string.len = slen;
                memcpy(heap[sptr].string.data, buf, slen + 1);
                PUSH(((Value){.type=VAL_STRING,.as.ptr=sptr}));
                break;
            }
            case 54: { /* string-append: pop 2 strings, concat */
                Value b = POP(), a = POP();
                int32_t sptr = HALLOC(); if (sptr < 0) break;
                heap[sptr].type = HEAP_STRING;
                int la = (a.type==VAL_STRING) ? heap[a.as.ptr].string.len : 0;
                int lb = (b.type==VAL_STRING) ? heap[b.as.ptr].string.len : 0;
                if (la + lb > 255) { la = (la > 255) ? 255 : la; lb = 255 - la; }
                heap[sptr].string.len = la + lb;
                if (a.type==VAL_STRING) memcpy(heap[sptr].string.data, heap[a.as.ptr].string.data, la);
                if (b.type==VAL_STRING) memcpy(heap[sptr].string.data + la, heap[b.as.ptr].string.data, lb);
                heap[sptr].string.data[la+lb] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=sptr}));
                break;
            }
            case 55: { /* string=? */
                Value b = POP(), a = POP();
                int eq = 0;
                if (a.type==VAL_STRING && b.type==VAL_STRING) {
                    eq = (heap[a.as.ptr].string.len == heap[b.as.ptr].string.len &&
                          memcmp(heap[a.as.ptr].string.data, heap[b.as.ptr].string.data, heap[a.as.ptr].string.len) == 0);
                }
                PUSH(((Value){.type=VAL_BOOL,.as.b=eq}));
                break;
            }
            case 60: { /* newline */
                printf("\n"); break;
            }
            case 70: { /* apply: f args-list → call f with unpacked args */
                Value args_list = POP(), func = POP();
                if (func.type != VAL_CLOSURE) { printf("APPLY non-function\n"); error=1; break; }
                /* Count args */
                int argc = 0;
                Value tmp = args_list;
                while (tmp.type == VAL_PAIR) { argc++; tmp = heap[tmp.as.ptr].cons.cdr; }
                /* Push function below args */
                PUSH(func);
                /* Unpack args onto stack */
                tmp = args_list;
                while (tmp.type == VAL_PAIR) {
                    PUSH(heap[tmp.as.ptr].cons.car);
                    tmp = heap[tmp.as.ptr].cons.cdr;
                }
                /* Now stack: [..., func, arg0, arg1, ...]. Call. */
                if (frame_count >= MAX_FRAMES) { error=1; break; }
                frames[frame_count].return_pc = pc;
                frames[frame_count].return_fp = fp;
                frames[frame_count].heap_mark = heap_next;
                frames[frame_count].force_promise_ptr = -1;
                frame_count++;
                fp = sp - argc;
                pc = heap[func.as.ptr].closure.func_pc;
                break;
            }
            case 56: { /* string-length */
                Value v=POP();
                if (v.type==VAL_STRING) PUSH(INT_VAL(heap[v.as.ptr].string.len));
                else PUSH(INT_VAL(0));
                break;
            }
            case 57: { /* string-ref */
                Value idx=POP(), s=POP();
                if (s.type==VAL_STRING) {
                    int i=(int)AS_NUM(idx);
                    if(i>=0&&i<heap[s.as.ptr].string.len) PUSH(INT_VAL(heap[s.as.ptr].string.data[i]));
                    else PUSH(INT_VAL(0));
                } else PUSH(INT_VAL(0));
                break;
            }
            case 71: { /* length: list → int */
                Value lst = POP();
                int len = 0;
                while (lst.type == VAL_PAIR) { len++; lst = heap[lst.as.ptr].cons.cdr; }
                PUSH(INT_VAL(len));
                break;
            }
            case 72: { /* car */
                Value p=POP();
                if(p.type==VAL_PAIR) PUSH(heap[p.as.ptr].cons.car);
                else { printf("CAR non-pair\n"); PUSH(((Value){.type=VAL_NIL})); }
                break;
            }
            case 73: { /* cdr */
                Value p=POP();
                if(p.type==VAL_PAIR) PUSH(heap[p.as.ptr].cons.cdr);
                else { printf("CDR non-pair\n"); PUSH(((Value){.type=VAL_NIL})); }
                break;
            }
            case 74: { /* cons */
                Value cdr_v=POP(), car_v=POP();
                int32_t ptr=HALLOC(); if(ptr<0) break;
                heap[ptr].type=HEAP_CONS;
                heap[ptr].cons.car=car_v; heap[ptr].cons.cdr=cdr_v;
                PUSH(((Value){.type=VAL_PAIR,.as.ptr=ptr}));
                break;
            }
            case 75: { /* set-car! */ Value v=POP(),p=POP(); if(p.type==VAL_PAIR) heap[p.as.ptr].cons.car=v; break; }
            case 76: { /* set-cdr! */ Value v=POP(),p=POP(); if(p.type==VAL_PAIR) heap[p.as.ptr].cons.cdr=v; break; }
            case 77: { /* cadr */ Value p=POP(); if(p.type==VAL_PAIR) { Value c=heap[p.as.ptr].cons.cdr; if(c.type==VAL_PAIR) PUSH(heap[c.as.ptr].cons.car); else PUSH(((Value){.type=VAL_NIL})); } else PUSH(((Value){.type=VAL_NIL})); break; }
            case 78: { /* cddr */ Value p=POP(); if(p.type==VAL_PAIR) { Value c=heap[p.as.ptr].cons.cdr; if(c.type==VAL_PAIR) PUSH(heap[c.as.ptr].cons.cdr); else PUSH(((Value){.type=VAL_NIL})); } else PUSH(((Value){.type=VAL_NIL})); break; }
            case 79: { /* caar */ Value p=POP(); if(p.type==VAL_PAIR) { Value c=heap[p.as.ptr].cons.car; if(c.type==VAL_PAIR) PUSH(heap[c.as.ptr].cons.car); else PUSH(((Value){.type=VAL_NIL})); } else PUSH(((Value){.type=VAL_NIL})); break; }
            case 80: { /* caddr */ Value p=POP(); if(p.type==VAL_PAIR) { Value c1=heap[p.as.ptr].cons.cdr; if(c1.type==VAL_PAIR) { Value c2=heap[c1.as.ptr].cons.cdr; if(c2.type==VAL_PAIR) PUSH(heap[c2.as.ptr].cons.car); else PUSH(((Value){.type=VAL_NIL})); } else PUSH(((Value){.type=VAL_NIL})); } else PUSH(((Value){.type=VAL_NIL})); break; }
            case 81: { /* vector-ref */ Value idx=POP(),v=POP(); if(v.type==VAL_VECTOR){int i=(int)AS_NUM(idx); if(i>=0&&i<heap[v.as.ptr].vector.len) PUSH(heap[v.as.ptr].vector.items[i]); else PUSH(((Value){.type=VAL_NIL}));} else PUSH(((Value){.type=VAL_NIL})); break; }
            case 82: { /* vector-length */ Value v=POP(); if(v.type==VAL_VECTOR) PUSH(INT_VAL(heap[v.as.ptr].vector.len)); else PUSH(INT_VAL(0)); break; }
            case 83: { /* vector-set! */ Value val=POP(),idx=POP(),v=POP(); if(v.type==VAL_VECTOR){int i=(int)AS_NUM(idx); if(i>=0&&i<heap[v.as.ptr].vector.len) heap[v.as.ptr].vector.items[i]=val;} break; }
            case 90: /* display — same as PRINT but without "→ " prefix */
            case 91: { /* write */
                Value v = POP();
                if (v.type==VAL_INT) printf("%lld", (long long)v.as.i);
                else if (v.type==VAL_FLOAT) printf("%.6g", v.as.f);
                else if (v.type==VAL_BOOL) printf("%s", v.as.b?"#t":"#f");
                else if (v.type==VAL_NIL) printf("()");
                else if (v.type==VAL_PAIR) {
                    printf("("); Value cur=v; int f=1;
                    while(cur.type==VAL_PAIR){if(!f)printf(" ");f=0;
                    Value car=heap[cur.as.ptr].cons.car;
                    if(car.type==VAL_INT)printf("%lld",(long long)car.as.i);
                    else if(car.type==VAL_FLOAT)printf("%.6g",car.as.f);
                    else if(car.type==VAL_BOOL)printf("%s",car.as.b?"#t":"#f");
                    else if(car.type==VAL_NIL)printf("()");
                    else printf("<v>");
                    cur=heap[cur.as.ptr].cons.cdr;}
                    if(cur.type!=VAL_NIL){printf(" . ");if(cur.type==VAL_INT)printf("%lld",(long long)cur.as.i);else printf("<v>");}
                    printf(")");
                }
                else if (v.type==VAL_CLOSURE) printf("<procedure>");
                else if (v.type==VAL_STRING) printf("%.*s", heap[v.as.ptr].string.len, heap[v.as.ptr].string.data);
                else if (v.type==VAL_VECTOR) {
                    printf("#("); for(int vi=0;vi<heap[v.as.ptr].vector.len;vi++){if(vi)printf(" ");Value it=heap[v.as.ptr].vector.items[vi];if(it.type==VAL_INT)printf("%lld",(long long)it.as.i);else if(it.type==VAL_FLOAT)printf("%.6g",it.as.f);else printf("<v>");} printf(")");
                }
                printf("\n");
                PUSH(((Value){.type=VAL_NIL})); /* display returns void */
                break;
            }
            case 130: { /* raise: throw exception value to nearest handler */
                Value exn = POP();
                if (handler_count <= 0) {
                    printf("ERROR: unhandled exception: ");
                    if (exn.type == VAL_INT) printf("%lld", (long long)exn.as.i);
                    else if (exn.type == VAL_FLOAT) printf("%.6g", exn.as.f);
                    else if (exn.type == VAL_STRING) printf("%.*s", heap[exn.as.ptr].string.len, heap[exn.as.ptr].string.data);
                    else if (exn.type == VAL_BOOL) printf("%s", exn.as.b ? "#t" : "#f");
                    else printf("<value>");
                    printf("\n");
                    error = 1; break;
                }
                handler_count--;
                /* Unwind to handler point — do NOT restore stack values.
                 * This preserves set! side effects (R7RS semantics).
                 * Only restore sp/fp/frame_count/pc. */
                sp = exc_handlers[handler_count].saved_sp;
                fp = exc_handlers[handler_count].saved_fp;
                frame_count = exc_handlers[handler_count].saved_frame_count;
                pc = exc_handlers[handler_count].handler_pc;
                /* Store exception in VM register — handler accesses via OP_GET_EXN */
                current_exn = exn;
                break;
            }
            case 133: { /* eq?: identity equality (pointer for heap types, value for scalars) */
                Value b=POP(), a=POP();
                int result = 0;
                if (a.type != b.type) result = 0;
                else if (a.type == VAL_NIL) result = 1;
                else if (a.type == VAL_BOOL) result = (a.as.b == b.as.b);
                else if (a.type == VAL_INT) result = (a.as.i == b.as.i);
                else if (a.type == VAL_FLOAT) result = (a.as.f == b.as.f);
                else result = (a.as.ptr == b.as.ptr); /* pointer equality for heap types */
                PUSH(((Value){.type=VAL_BOOL,.as.b=result}));
                break;
            }
            case 134: { /* equal?: deep structural equality */
                Value b=POP(), a=POP();
                /* Iterative deep comparison using a work stack */
                Value work_a[64], work_b[64]; int wn = 0;
                work_a[wn] = a; work_b[wn] = b; wn++;
                int result = 1;
                while (wn > 0 && result) {
                    wn--;
                    Value x = work_a[wn], y = work_b[wn];
                    if (x.type != y.type) { result = 0; break; }
                    if (x.type == VAL_NIL) continue;
                    else if (x.type == VAL_BOOL) { if (x.as.b != y.as.b) result = 0; }
                    else if (x.type == VAL_INT) { if (x.as.i != y.as.i) result = 0; }
                    else if (x.type == VAL_FLOAT) { if (x.as.f != y.as.f) result = 0; }
                    else if (x.type == VAL_STRING) {
                        if (heap[x.as.ptr].string.len != heap[y.as.ptr].string.len) result = 0;
                        else if (memcmp(heap[x.as.ptr].string.data, heap[y.as.ptr].string.data,
                                        heap[x.as.ptr].string.len) != 0) result = 0;
                    }
                    else if (x.type == VAL_PAIR) {
                        if (wn + 2 > 64) { result = (x.as.ptr == y.as.ptr); } /* overflow: fallback */
                        else {
                            work_a[wn] = heap[x.as.ptr].cons.cdr; work_b[wn] = heap[y.as.ptr].cons.cdr; wn++;
                            work_a[wn] = heap[x.as.ptr].cons.car; work_b[wn] = heap[y.as.ptr].cons.car; wn++;
                        }
                    }
                    else if (x.type == VAL_VECTOR) {
                        int la = heap[x.as.ptr].vector.len, lb = heap[y.as.ptr].vector.len;
                        if (la != lb) result = 0;
                        else { for (int vi=0; vi < la && wn < 64; vi++) {
                            work_a[wn] = heap[x.as.ptr].vector.items[vi];
                            work_b[wn] = heap[y.as.ptr].vector.items[vi]; wn++;
                        }}
                    }
                    else result = (x.as.ptr == y.as.ptr);
                }
                PUSH(((Value){.type=VAL_BOOL,.as.b=result}));
                break;
            }
            case 135: { /* append: append two lists */
                Value b=POP(), a=POP();
                if (a.type == VAL_NIL) { PUSH(b); break; }
                if (a.type != VAL_PAIR) { PUSH(b); break; }
                /* Copy list a, set last cdr to b */
                Value head_v = {.type=VAL_NIL}, tail_v = {.type=VAL_NIL};
                Value cur = a;
                while (cur.type == VAL_PAIR) {
                    int32_t ptr = HALLOC(); if (ptr < 0) goto append_done;
                    heap[ptr].type = HEAP_CONS;
                    heap[ptr].cons.car = heap[cur.as.ptr].cons.car;
                    heap[ptr].cons.cdr = (Value){.type=VAL_NIL};
                    Value node_v = {.type=VAL_PAIR,.as.ptr=ptr};
                    if (head_v.type == VAL_NIL) head_v = node_v;
                    else heap[tail_v.as.ptr].cons.cdr = node_v;
                    tail_v = node_v;
                    cur = heap[cur.as.ptr].cons.cdr;
                }
                if (tail_v.type == VAL_PAIR) heap[tail_v.as.ptr].cons.cdr = b;
                append_done:
                PUSH(head_v);
                break;
            }
            case 136: { /* reverse: reverse a list */
                Value lst = POP();
                Value result_v = {.type=VAL_NIL};
                while (lst.type == VAL_PAIR) {
                    int32_t ptr = HALLOC(); if (ptr < 0) break;
                    heap[ptr].type = HEAP_CONS;
                    heap[ptr].cons.car = heap[lst.as.ptr].cons.car;
                    heap[ptr].cons.cdr = result_v;
                    result_v = (Value){.type=VAL_PAIR,.as.ptr=ptr};
                    lst = heap[lst.as.ptr].cons.cdr;
                }
                PUSH(result_v);
                break;
            }
            case 137: { /* member: (member val lst) → sublist starting at val, or #f */
                Value lst=POP(), val=POP();
                Value cur = lst;
                while (cur.type == VAL_PAIR) {
                    Value car = heap[cur.as.ptr].cons.car;
                    /* Use equal? semantics: numeric or pointer equality */
                    int match = 0;
                    if (car.type == val.type) {
                        if (car.type == VAL_INT) match = (car.as.i == val.as.i);
                        else if (car.type == VAL_FLOAT) match = (car.as.f == val.as.f);
                        else if (car.type == VAL_BOOL) match = (car.as.b == val.as.b);
                        else if (car.type == VAL_NIL) match = 1;
                        else match = (car.as.ptr == val.as.ptr);
                    }
                    if (match) { PUSH(cur); goto member_done; }
                    cur = heap[cur.as.ptr].cons.cdr;
                }
                PUSH(((Value){.type=VAL_BOOL,.as.b=0}));
                member_done: break;
            }
            case 138: { /* assoc: (assoc key alist) → pair or #f */
                Value alist=POP(), key=POP();
                Value cur = alist;
                while (cur.type == VAL_PAIR) {
                    Value pair = heap[cur.as.ptr].cons.car;
                    if (pair.type == VAL_PAIR) {
                        Value k = heap[pair.as.ptr].cons.car;
                        int match = 0;
                        if (k.type == key.type) {
                            if (k.type == VAL_INT) match = (k.as.i == key.as.i);
                            else if (k.type == VAL_FLOAT) match = (k.as.f == key.as.f);
                            else if (k.type == VAL_STRING)
                                match = (heap[k.as.ptr].string.len == heap[key.as.ptr].string.len &&
                                         memcmp(heap[k.as.ptr].string.data,
                                                heap[key.as.ptr].string.data,
                                                heap[k.as.ptr].string.len) == 0);
                            else match = (k.as.ptr == key.as.ptr);
                        }
                        if (match) { PUSH(pair); goto assoc_done; }
                    }
                    cur = heap[cur.as.ptr].cons.cdr;
                }
                PUSH(((Value){.type=VAL_BOOL,.as.b=0}));
                assoc_done: break;
            }
            case 139: { /* list->vector */
                Value lst=POP();
                int len = 0; Value tmp = lst;
                while (tmp.type == VAL_PAIR) { len++; tmp = heap[tmp.as.ptr].cons.cdr; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = len < 64 ? len : 64;
                tmp = lst;
                for (int vi=0; vi < len && vi < 64; vi++) {
                    heap[ptr].vector.items[vi] = heap[tmp.as.ptr].cons.car;
                    tmp = heap[tmp.as.ptr].cons.cdr;
                }
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 140: { /* vector->list */
                Value v=POP();
                if (v.type != VAL_VECTOR) { PUSH(((Value){.type=VAL_NIL})); break; }
                Value result_v = {.type=VAL_NIL};
                for (int vi = heap[v.as.ptr].vector.len - 1; vi >= 0; vi--) {
                    int32_t ptr = HALLOC(); if (ptr < 0) break;
                    heap[ptr].type = HEAP_CONS;
                    heap[ptr].cons.car = heap[v.as.ptr].vector.items[vi];
                    heap[ptr].cons.cdr = result_v;
                    result_v = (Value){.type=VAL_PAIR,.as.ptr=ptr};
                }
                PUSH(result_v);
                break;
            }
            case 142: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)+AS_NUM(b))); break; } /* + */
            case 143: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)-AS_NUM(b))); break; } /* - */
            case 144: { Value b=POP(),a=POP(); PUSH(NUM_VAL(AS_NUM(a)*AS_NUM(b))); break; } /* * */
            case 145: { Value b=POP(),a=POP(); double bv=AS_NUM(b); PUSH(NUM_VAL(AS_NUM(a)/bv)); break; } /* / */
            case 141: { /* iota: (iota n) → (0 1 2 ... n-1) */
                Value n_v=POP();
                int n = (int)AS_NUM(n_v);
                Value result_v = {.type=VAL_NIL};
                for (int i = n-1; i >= 0; i--) {
                    int32_t ptr = HALLOC(); if (ptr < 0) break;
                    heap[ptr].type = HEAP_CONS;
                    heap[ptr].cons.car = INT_VAL(i);
                    heap[ptr].cons.cdr = result_v;
                    result_v = (Value){.type=VAL_PAIR,.as.ptr=ptr};
                }
                PUSH(result_v);
                break;
            }
            /* === Additional predicates === */
            case 160: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_STRING)})); break; } /* symbol? (strings are symbols in our VM) */
            case 161: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_INT && v.as.i >= 0 && v.as.i < 128)})); break; } /* char? */
            case 162: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_INT)})); break; } /* exact? */
            case 163: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_FLOAT)})); break; } /* inexact? */
            case 164: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_FLOAT && isnan(v.as.f))})); break; } /* nan? */
            case 165: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_FLOAT && isinf(v.as.f))})); break; } /* infinite? */
            case 166: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=((v.type==VAL_INT) || (v.type==VAL_FLOAT && isfinite(v.as.f)))})); break; } /* finite? */

            /* === String operations === */
            case 170: { /* substring: str, start, end → new string */
                Value end_v=POP(), start_v=POP(), s=POP();
                int start = (int)AS_NUM(start_v), end = (int)AS_NUM(end_v);
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int slen = heap[s.as.ptr].string.len;
                if (start < 0) start = 0; if (end > slen) end = slen;
                int nlen = end - start; if (nlen < 0) nlen = 0; if (nlen > 255) nlen = 255;
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                heap[ptr].string.len = nlen;
                memcpy(heap[ptr].string.data, heap[s.as.ptr].string.data + start, nlen);
                heap[ptr].string.data[nlen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr}));
                break;
            }
            case 171: { /* string-contains: str, substr → index or #f */
                Value sub=POP(), s=POP();
                if (s.type != VAL_STRING || sub.type != VAL_STRING) { PUSH(((Value){.type=VAL_BOOL,.as.b=0})); break; }
                char* found = strstr(heap[s.as.ptr].string.data, heap[sub.as.ptr].string.data);
                if (found) PUSH(INT_VAL(found - heap[s.as.ptr].string.data));
                else PUSH(((Value){.type=VAL_BOOL,.as.b=0}));
                break;
            }
            case 172: { /* string-upcase */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                int slen = heap[s.as.ptr].string.len; if (slen > 255) slen = 255;
                heap[ptr].string.len = slen;
                for (int i = 0; i < slen; i++) heap[ptr].string.data[i] = toupper(heap[s.as.ptr].string.data[i]);
                heap[ptr].string.data[slen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr}));
                break;
            }
            case 173: { /* string-downcase */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                int slen = heap[s.as.ptr].string.len; if (slen > 255) slen = 255;
                heap[ptr].string.len = slen;
                for (int i = 0; i < slen; i++) heap[ptr].string.data[i] = tolower(heap[s.as.ptr].string.data[i]);
                heap[ptr].string.data[slen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr}));
                break;
            }
            case 174: { /* string-reverse */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                int slen = heap[s.as.ptr].string.len; if (slen > 255) slen = 255;
                heap[ptr].string.len = slen;
                for (int i = 0; i < slen; i++) heap[ptr].string.data[i] = heap[s.as.ptr].string.data[slen - 1 - i];
                heap[ptr].string.data[slen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr}));
                break;
            }
            case 175: { /* string->number */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(((Value){.type=VAL_BOOL,.as.b=0})); break; }
                char* endp;
                double v = strtod(heap[s.as.ptr].string.data, &endp);
                if (endp == heap[s.as.ptr].string.data) PUSH(((Value){.type=VAL_BOOL,.as.b=0}));
                else PUSH(NUM_VAL(v));
                break;
            }
            case 176: { /* string->list: convert string to list of char codes */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(((Value){.type=VAL_NIL})); break; }
                Value result = {.type=VAL_NIL};
                for (int i = heap[s.as.ptr].string.len - 1; i >= 0; i--) {
                    int32_t p = HALLOC(); if (p < 0) break;
                    heap[p].type = HEAP_CONS;
                    heap[p].cons.car = INT_VAL((unsigned char)heap[s.as.ptr].string.data[i]);
                    heap[p].cons.cdr = result;
                    result = (Value){.type=VAL_PAIR,.as.ptr=p};
                }
                PUSH(result); break;
            }
            case 177: { /* list->string: convert list of char codes to string */
                Value lst=POP();
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                int len = 0; Value cur = lst;
                while (cur.type == VAL_PAIR && len < 255) {
                    Value ch = heap[cur.as.ptr].cons.car;
                    heap[ptr].string.data[len++] = (char)(int)AS_NUM(ch);
                    cur = heap[cur.as.ptr].cons.cdr;
                }
                heap[ptr].string.len = len;
                heap[ptr].string.data[len] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr})); break;
            }
            case 178: { /* string-copy */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING;
                heap[ptr].string.len = heap[s.as.ptr].string.len;
                memcpy(heap[ptr].string.data, heap[s.as.ptr].string.data, heap[s.as.ptr].string.len + 1);
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr})); break;
            }
            case 180: { Value v=POP(); PUSH(FLOAT_VAL(AS_NUM(v))); break; } /* exact->inexact */
            case 181: { Value v=POP(); PUSH(INT_VAL((int64_t)AS_NUM(v))); break; } /* inexact->exact */
            case 182: { Value v=POP(); PUSH(INT_VAL((int64_t)AS_NUM(v))); break; } /* char->integer */
            case 183: { Value v=POP(); PUSH(INT_VAL((int64_t)AS_NUM(v))); break; } /* integer->char */
            case 184: { /* symbol->string (symbols ARE strings in our VM, just return) */
                /* Top of stack is the symbol (string). Just leave it. */
                break;
            }
            case 185: { /* string->symbol: symbols ARE strings in this VM, passthrough */ break; }
            case 186: { /* list-ref: lst, index → element */
                Value idx_v=POP(), lst=POP();
                int idx = (int)AS_NUM(idx_v);
                while (lst.type == VAL_PAIR && idx > 0) { lst = heap[lst.as.ptr].cons.cdr; idx--; }
                if (lst.type == VAL_PAIR) PUSH(heap[lst.as.ptr].cons.car);
                else PUSH(((Value){.type=VAL_NIL}));
                break;
            }
            case 187: { /* list-tail: lst, k → sublist */
                Value k_v=POP(), lst=POP();
                int k = (int)AS_NUM(k_v);
                while (lst.type == VAL_PAIR && k > 0) { lst = heap[lst.as.ptr].cons.cdr; k--; }
                PUSH(lst); break;
            }
            case 188: { /* last-pair */
                Value lst=POP();
                if (lst.type != VAL_PAIR) { PUSH(lst); break; }
                while (heap[lst.as.ptr].cons.cdr.type == VAL_PAIR)
                    lst = heap[lst.as.ptr].cons.cdr;
                PUSH(lst); break;
            }
            case 189: { /* list? */
                Value lst=POP();
                int is_list = 1;
                Value cur = lst;
                int count = 0;
                while (cur.type == VAL_PAIR && count < 10000) { cur = heap[cur.as.ptr].cons.cdr; count++; }
                is_list = (cur.type == VAL_NIL);
                PUSH(((Value){.type=VAL_BOOL,.as.b=is_list})); break;
            }
            case 190: { Value v=POP(); PUSH(NUM_VAL(trunc(AS_NUM(v)))); break; } /* truncate */
            case 191: { Value v=POP(); PUSH(INT_VAL((int64_t)AS_NUM(v))); break; } /* exact */
            case 192: { Value v=POP(); PUSH(FLOAT_VAL(AS_NUM(v))); break; } /* inexact */

            /* === Hash tables === */
            case 200: { /* make-hash-table */
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_HASH;
                heap[ptr].hash.count = 0;
                PUSH(((Value){.type=VAL_HASH,.as.ptr=ptr}));
                break;
            }
            case 201: { /* hash-ref: hash key → value or #f */
                Value key=POP(), h=POP();
                if (h.type != VAL_HASH) { PUSH(((Value){.type=VAL_BOOL,.as.b=0})); break; }
                int found_it = 0;
                for (int hi = 0; hi < heap[h.as.ptr].hash.count; hi++) {
                    Value k = heap[h.as.ptr].hash.keys[hi];
                    int match = 0;
                    if (k.type == key.type) {
                        if (k.type == VAL_INT) match = (k.as.i == key.as.i);
                        else if (k.type == VAL_FLOAT) match = (k.as.f == key.as.f);
                        else if (k.type == VAL_STRING) match = (heap[k.as.ptr].string.len == heap[key.as.ptr].string.len && memcmp(heap[k.as.ptr].string.data, heap[key.as.ptr].string.data, heap[k.as.ptr].string.len) == 0);
                        else match = (k.as.ptr == key.as.ptr);
                    }
                    if (match) { PUSH(heap[h.as.ptr].hash.vals[hi]); found_it = 1; break; }
                }
                if (!found_it) PUSH(((Value){.type=VAL_BOOL,.as.b=0}));
                break;
            }
            case 202: { /* hash-set!: hash key value → void */
                Value val=POP(), key=POP(), h=POP();
                if (h.type != VAL_HASH) break;
                /* Check if key exists */
                int found_it = 0;
                for (int hi = 0; hi < heap[h.as.ptr].hash.count; hi++) {
                    Value k = heap[h.as.ptr].hash.keys[hi];
                    int match = 0;
                    if (k.type == key.type) {
                        if (k.type == VAL_INT) match = (k.as.i == key.as.i);
                        else if (k.type == VAL_FLOAT) match = (k.as.f == key.as.f);
                        else if (k.type == VAL_STRING) match = (heap[k.as.ptr].string.len == heap[key.as.ptr].string.len && memcmp(heap[k.as.ptr].string.data, heap[key.as.ptr].string.data, heap[k.as.ptr].string.len) == 0);
                        else match = (k.as.ptr == key.as.ptr);
                    }
                    if (match) { heap[h.as.ptr].hash.vals[hi] = val; found_it = 1; break; }
                }
                if (!found_it && heap[h.as.ptr].hash.count < 32) {
                    int idx = heap[h.as.ptr].hash.count++;
                    heap[h.as.ptr].hash.keys[idx] = key;
                    heap[h.as.ptr].hash.vals[idx] = val;
                }
                PUSH(((Value){.type=VAL_NIL}));
                break;
            }
            case 203: { /* hash-has-key? */
                Value key=POP(), h=POP();
                if (h.type != VAL_HASH) { PUSH(((Value){.type=VAL_BOOL,.as.b=0})); break; }
                int found_it = 0;
                for (int hi = 0; hi < heap[h.as.ptr].hash.count && !found_it; hi++) {
                    Value k = heap[h.as.ptr].hash.keys[hi];
                    if (k.type == key.type) {
                        if (k.type == VAL_INT) found_it = (k.as.i == key.as.i);
                        else if (k.type == VAL_STRING) found_it = (heap[k.as.ptr].string.len == heap[key.as.ptr].string.len && memcmp(heap[k.as.ptr].string.data, heap[key.as.ptr].string.data, heap[k.as.ptr].string.len) == 0);
                        else found_it = (k.as.ptr == key.as.ptr);
                    }
                }
                PUSH(((Value){.type=VAL_BOOL,.as.b=found_it})); break;
            }
            case 204: { /* hash-keys → list of keys */
                Value h=POP();
                Value result = {.type=VAL_NIL};
                if (h.type == VAL_HASH) {
                    for (int hi = heap[h.as.ptr].hash.count - 1; hi >= 0; hi--) {
                        int32_t p = HALLOC(); if (p < 0) break;
                        heap[p].type = HEAP_CONS; heap[p].cons.car = heap[h.as.ptr].hash.keys[hi]; heap[p].cons.cdr = result;
                        result = (Value){.type=VAL_PAIR,.as.ptr=p};
                    }
                }
                PUSH(result); break;
            }
            case 205: { /* hash-values → list of values */
                Value h=POP();
                Value result = {.type=VAL_NIL};
                if (h.type == VAL_HASH) {
                    for (int hi = heap[h.as.ptr].hash.count - 1; hi >= 0; hi--) {
                        int32_t p = HALLOC(); if (p < 0) break;
                        heap[p].type = HEAP_CONS; heap[p].cons.car = heap[h.as.ptr].hash.vals[hi]; heap[p].cons.cdr = result;
                        result = (Value){.type=VAL_PAIR,.as.ptr=p};
                    }
                }
                PUSH(result); break;
            }
            case 206: { /* hash-count */
                Value h=POP();
                PUSH(INT_VAL(h.type == VAL_HASH ? heap[h.as.ptr].hash.count : 0)); break;
            }
            case 207: { /* hash-delete! */
                Value key=POP(), h=POP();
                if (h.type == VAL_HASH) {
                    for (int hi = 0; hi < heap[h.as.ptr].hash.count; hi++) {
                        Value k = heap[h.as.ptr].hash.keys[hi];
                        int match = (k.type == key.type && ((k.type == VAL_INT && k.as.i == key.as.i) || (k.type == VAL_STRING && heap[k.as.ptr].string.len == heap[key.as.ptr].string.len && memcmp(heap[k.as.ptr].string.data, heap[key.as.ptr].string.data, heap[k.as.ptr].string.len) == 0)));
                        if (match) {
                            if (heap[h.as.ptr].hash.count > 0) {
                                heap[h.as.ptr].hash.count--;
                                heap[h.as.ptr].hash.keys[hi] = heap[h.as.ptr].hash.keys[heap[h.as.ptr].hash.count];
                                heap[h.as.ptr].hash.vals[hi] = heap[h.as.ptr].hash.vals[heap[h.as.ptr].hash.count];
                            }
                            break;
                        }
                    }
                }
                PUSH(((Value){.type=VAL_NIL})); break;
            }

            /* === Characters === */
            case 210: { Value v=POP(); int c2=(int)AS_NUM(v); PUSH(((Value){.type=VAL_BOOL,.as.b=isalpha(c2)})); break; } /* char-alphabetic? */
            case 211: { Value v=POP(); int c2=(int)AS_NUM(v); PUSH(((Value){.type=VAL_BOOL,.as.b=isdigit(c2)})); break; } /* char-numeric? */
            case 212: { Value v=POP(); int c2=(int)AS_NUM(v); PUSH(((Value){.type=VAL_BOOL,.as.b=isspace(c2)})); break; } /* char-whitespace? */
            case 213: { Value v=POP(); int c2=(int)AS_NUM(v); PUSH(((Value){.type=VAL_BOOL,.as.b=isupper(c2)})); break; } /* char-upper-case? */
            case 214: { Value v=POP(); int c2=(int)AS_NUM(v); PUSH(((Value){.type=VAL_BOOL,.as.b=islower(c2)})); break; } /* char-lower-case? */
            case 215: { Value v=POP(); PUSH(INT_VAL(toupper((int)AS_NUM(v)))); break; } /* char-upcase */
            case 216: { Value v=POP(); PUSH(INT_VAL(tolower((int)AS_NUM(v)))); break; } /* char-downcase */
            case 217: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=((int)AS_NUM(a)==(int)AS_NUM(b))})); break; } /* char=? */
            case 218: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=((int)AS_NUM(a)<(int)AS_NUM(b))})); break; } /* char<? */
            case 219: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=((int)AS_NUM(a)>(int)AS_NUM(b))})); break; } /* char>? */

            /* === Bitwise === */
            case 220: { Value b=POP(),a=POP(); PUSH(INT_VAL((int64_t)AS_NUM(a) & (int64_t)AS_NUM(b))); break; } /* bitwise-and */
            case 221: { Value b=POP(),a=POP(); PUSH(INT_VAL((int64_t)AS_NUM(a) | (int64_t)AS_NUM(b))); break; } /* bitwise-or */
            case 222: { Value b=POP(),a=POP(); PUSH(INT_VAL((int64_t)AS_NUM(a) ^ (int64_t)AS_NUM(b))); break; } /* bitwise-xor */
            case 223: { Value a=POP(); PUSH(INT_VAL(~(int64_t)AS_NUM(a))); break; } /* bitwise-not */
            case 224: { Value b=POP(),a=POP(); int64_t n=(int64_t)AS_NUM(b); PUSH(INT_VAL(n>=0 ? ((int64_t)AS_NUM(a)<<n) : ((int64_t)AS_NUM(a)>>(-n)))); break; } /* arithmetic-shift */

            /* === Additional list ops (these call closures, implemented in Scheme prelude) */
            /* take(225), drop(226), any(227), every(228), find(229), sort(230) */
            /* These are handled by the Scheme prelude definitions above. */
            /* Native stubs kept as fallback — should not be reached. */
            case 225: case 226: case 227: case 228: case 229: case 230:
                fprintf(stderr, "NATIVE_CALL %d: use Scheme prelude version\n", fid);
                PUSH(((Value){.type=VAL_NIL})); break;

            /* === Additional string ops === */
            case 231: { /* string-repeat: str n → repeated string */
                Value n_v=POP(), s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int n = (int)AS_NUM(n_v);
                int slen = heap[s.as.ptr].string.len;
                /* P1: guard negative n (rlen<0 → OOB write at data[rlen]),
                   slen==0 (i%slen div-by-zero), and slen*n int overflow. The
                   inline string.data buffer is 256 bytes, so clamp rlen to 255. */
                if (n < 0 || slen <= 0) { PUSH(s); break; }
                int64_t rlen64 = (int64_t)slen * (int64_t)n;
                int rlen = rlen64 > 255 ? 255 : (int)rlen64;
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING; heap[ptr].string.len = rlen;
                for (int i = 0; i < rlen; i++) heap[ptr].string.data[i] = heap[s.as.ptr].string.data[i % slen];
                heap[ptr].string.data[rlen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr})); break;
            }
            case 232: { /* string-trim */
                Value s=POP();
                if (s.type != VAL_STRING) { PUSH(s); break; }
                int slen = heap[s.as.ptr].string.len;
                int start = 0, end = slen;
                while (start < end && isspace(heap[s.as.ptr].string.data[start])) start++;
                while (end > start && isspace(heap[s.as.ptr].string.data[end-1])) end--;
                int nlen = end - start;
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING; heap[ptr].string.len = nlen;
                memcpy(heap[ptr].string.data, heap[s.as.ptr].string.data + start, nlen);
                heap[ptr].string.data[nlen] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr})); break;
            }
            case 233: { /* string-split: str delim → list of strings */
                Value delim=POP(), s=POP();
                if (s.type != VAL_STRING || delim.type != VAL_STRING) { PUSH(((Value){.type=VAL_NIL})); break; }
                Value result = {.type=VAL_NIL};
                char* str = heap[s.as.ptr].string.data;
                int dlen = heap[delim.as.ptr].string.len;
                char* d = heap[delim.as.ptr].string.data;
                /* Simple split */
                char* prev = str;
                for (char* p = str; *p; p++) {
                    if (dlen > 0 && strncmp(p, d, dlen) == 0) {
                        int seglen = (int)(p - prev);
                        int32_t sptr = HALLOC(); if (sptr < 0) break;
                        heap[sptr].type = HEAP_STRING; heap[sptr].string.len = seglen;
                        memcpy(heap[sptr].string.data, prev, seglen); heap[sptr].string.data[seglen] = 0;
                        int32_t cp = HALLOC(); if (cp < 0) break;
                        heap[cp].type = HEAP_CONS;
                        heap[cp].cons.car = (Value){.type=VAL_STRING,.as.ptr=sptr};
                        heap[cp].cons.cdr = result;
                        result = (Value){.type=VAL_PAIR,.as.ptr=cp};
                        p += dlen - 1; prev = p + 1;
                    }
                }
                /* Last segment */
                int seglen = (int)(str + heap[s.as.ptr].string.len - prev);
                int32_t sptr = HALLOC(); if (sptr < 0) { PUSH(result); break; }
                heap[sptr].type = HEAP_STRING; heap[sptr].string.len = seglen;
                memcpy(heap[sptr].string.data, prev, seglen); heap[sptr].string.data[seglen] = 0;
                int32_t cp = HALLOC(); if (cp < 0) { PUSH(result); break; }
                heap[cp].type = HEAP_CONS;
                heap[cp].cons.car = (Value){.type=VAL_STRING,.as.ptr=sptr};
                heap[cp].cons.cdr = result;
                result = (Value){.type=VAL_PAIR,.as.ptr=cp};
                /* Reverse (split built in reverse order) */
                Value rev = {.type=VAL_NIL};
                while (result.type == VAL_PAIR) {
                    int32_t rp = HALLOC(); if (rp < 0) break;
                    heap[rp].type = HEAP_CONS; heap[rp].cons.car = heap[result.as.ptr].cons.car; heap[rp].cons.cdr = rev;
                    rev = (Value){.type=VAL_PAIR,.as.ptr=rp};
                    result = heap[result.as.ptr].cons.cdr;
                }
                PUSH(rev); break;
            }
            case 234: { /* string-join: list-of-strings separator → string */
                Value sep=POP(), lst=POP();
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_STRING; heap[ptr].string.len = 0; heap[ptr].string.data[0] = 0;
                int pos = 0; int first_j = 1;
                Value cur = lst;
                while (cur.type == VAL_PAIR && pos < 250) {
                    Value s = heap[cur.as.ptr].cons.car;
                    if (!first_j && sep.type == VAL_STRING) {
                        int sl = heap[sep.as.ptr].string.len;
                        if (pos + sl > 255) { sl = 255 - pos; if (sl <= 0) break; }
                        memcpy(heap[ptr].string.data + pos, heap[sep.as.ptr].string.data, sl); pos += sl;
                    }
                    first_j = 0;
                    if (s.type == VAL_STRING) {
                        int sl = heap[s.as.ptr].string.len;
                        if (pos + sl > 255) { sl = 255 - pos; if (sl <= 0) break; }
                        memcpy(heap[ptr].string.data + pos, heap[s.as.ptr].string.data, sl); pos += sl;
                    }
                    cur = heap[cur.as.ptr].cons.cdr;
                }
                heap[ptr].string.len = pos; heap[ptr].string.data[pos] = 0;
                PUSH(((Value){.type=VAL_STRING,.as.ptr=ptr})); break;
            }

            /* === Misc === */
            case 235: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=IS_FALSY(v)})); break; } /* not */
            case 236: { Value b=POP(),a=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(a.as.b==b.as.b)})); break; } /* boolean=? */
            case 237: { /* error: raise with message */
                Value msg=POP(); PUSH(msg); /* push as exception value */
                /* Trigger raise logic */
                if (handler_count <= 0) {
                    printf("ERROR: "); if (msg.type==VAL_STRING) printf("%.*s",heap[msg.as.ptr].string.len,heap[msg.as.ptr].string.data); else printf("error"); printf("\n");
                    error=1; break;
                }
                handler_count--;
                sp = exc_handlers[handler_count].saved_sp;
                fp = exc_handlers[handler_count].saved_fp;
                frame_count = exc_handlers[handler_count].saved_frame_count;
                pc = exc_handlers[handler_count].handler_pc;
                current_exn = msg;
                break;
            }
            case 238: { PUSH(((Value){.type=VAL_NIL})); break; } /* void */
            case 239: { Value v=POP(); PUSH(((Value){.type=VAL_BOOL,.as.b=(v.type==VAL_HASH)})); break; } /* hash-table? */
            case 250: { Value x=POP(),y=POP(); PUSH(FLOAT_VAL(atan2(AS_NUM(y),AS_NUM(x)))); break; } /* atan2 */
            case 252: { /* propagate_open_slot: child_closure, child_uv_idx, parent_uv_idx → void
                         * If the parent closure (at stack[fp-1]) has an open slot for parent_uv_idx,
                         * copy that open slot to child_closure's child_uv_idx. */
                Value parent_uv = POP(), child_uv = POP(), child_cl = POP();
                int puv = (int)AS_NUM(parent_uv);
                int cuv = (int)AS_NUM(child_uv);
                Value parent_cl = stack[fp - 1]; /* current frame's closure */
                if (child_cl.type == VAL_CLOSURE && parent_cl.type == VAL_CLOSURE
                    && puv >= 0 && puv < heap[parent_cl.as.ptr].closure.n_upvalues
                    && cuv >= 0 && cuv < heap[child_cl.as.ptr].closure.n_upvalues) {
                    int32_t parent_open = heap[parent_cl.as.ptr].closure.open_slots[puv];
                    if (parent_open >= 0) {
                        /* Parent has an open slot — propagate to child */
                        heap[child_cl.as.ptr].closure.open_slots[cuv] = parent_open;
                    }
                }
                PUSH(((Value){.type=VAL_NIL})); break;
            }
            case 251: { /* call-with-values-apply: result consumer → call consumer with unpacked result */
                Value consumer=POP(), result=POP();
                if (consumer.type != VAL_CLOSURE) { PUSH(((Value){.type=VAL_NIL})); break; }
                if (result.type == VAL_VECTOR) {
                    /* Unpack vector values as arguments */
                    int argc = heap[result.as.ptr].vector.len;
                    PUSH(consumer);
                    for (int vi = 0; vi < argc; vi++)
                        PUSH(heap[result.as.ptr].vector.items[vi]);
                    if (frame_count >= MAX_FRAMES) { error=1; break; }
                    frames[frame_count].return_pc = pc;
                    frames[frame_count].return_fp = fp;
                    frames[frame_count].heap_mark = heap_next;
                    frames[frame_count].force_promise_ptr = -1;
                    frame_count++;
                    fp = sp - argc;
                    pc = heap[consumer.as.ptr].closure.func_pc;
                } else {
                    /* Single value — call consumer(result) */
                    PUSH(consumer);
                    PUSH(result);
                    if (frame_count >= MAX_FRAMES) { error=1; break; }
                    frames[frame_count].return_pc = pc;
                    frames[frame_count].return_fp = fp;
                    frames[frame_count].heap_mark = heap_next;
                    frames[frame_count].force_promise_ptr = -1;
                    frame_count++;
                    fp = sp - 1;
                    pc = heap[consumer.as.ptr].closure.func_pc;
                }
                break;
            }
            case 240: { /* display: human-readable output (no quotes on strings) */
                Value v=POP(); print_value(v, heap, 0, 0); PUSH(((Value){.type=VAL_NIL})); break;
            }
            case 241: { /* write: machine-readable output (strings quoted with escapes) */
                Value v=POP(); print_value(v, heap, 0, 1); PUSH(((Value){.type=VAL_NIL})); break;
            }

            case 151: { /* set_open_slot: closure, uv_idx, abs_slot → set one open upvalue slot */
                Value slot_v = POP(), idx_v = POP(), cl = POP();
                int idx = (int)AS_NUM(idx_v);
                int abs_slot = fp + (int)AS_NUM(slot_v); /* fp-relative → absolute */
                if (cl.type == VAL_CLOSURE && idx >= 0 && idx < heap[cl.as.ptr].closure.n_upvalues) {
                    heap[cl.as.ptr].closure.open_slots[idx] = abs_slot;
                }
                PUSH(((Value){.type=VAL_NIL}));
                break;
            }
            case 132: { /* force: force a promise #(forced? thunk-or-value) */
                Value prom = POP();
                if (prom.type != VAL_VECTOR || heap[prom.as.ptr].vector.len < 2) {
                    /* Not a promise — return as-is */
                    PUSH(prom);
                    break;
                }
                Value forced = heap[prom.as.ptr].vector.items[0];
                if (forced.type == VAL_BOOL && forced.as.b) {
                    /* Already forced — return cached value */
                    PUSH(heap[prom.as.ptr].vector.items[1]);
                } else {
                    /* Not yet forced — call the thunk */
                    Value thunk = heap[prom.as.ptr].vector.items[1];
                    if (thunk.type == VAL_CLOSURE) {
                        /* Call thunk inline: push closure, CALL 0 setup.
                         * Save promise index so we can memoize after return. */
                        PUSH(thunk);
                        if (frame_count >= MAX_FRAMES) { error=1; break; }
                        frames[frame_count].return_pc = pc;
                        frames[frame_count].return_fp = fp;
                        frames[frame_count].heap_mark = heap_next;
                        frames[frame_count].force_promise_ptr = prom.as.ptr;
                        frame_count++;
                        fp = sp;
                        pc = heap[thunk.as.ptr].closure.func_pc;
                    } else {
                        /* Thunk is already a value — memoize and return it */
                        heap[prom.as.ptr].vector.items[0] = (Value){.type=VAL_BOOL,.as.b=1};
                        PUSH(thunk);
                    }
                }
                break;
            }
            case 131: { /* open_upvalues: convert closure's upvalues to open (stack refs)
                         * Stack: [closure, count, base_slot]
                         * Sets closure.open_slots[i] = fp + base_slot + i for i in 0..count-1
                         * base_slot is compile-time relative; fp adjusts to runtime position */
                Value base_v = POP(), count_v = POP(), cl = POP();
                int count = (int)AS_NUM(count_v);
                int base = fp + (int)AS_NUM(base_v); /* fp-relative → absolute */
                if (cl.type == VAL_CLOSURE) {
                    for (int i = 0; i < count && i < heap[cl.as.ptr].closure.n_upvalues; i++) {
                        heap[cl.as.ptr].closure.open_slots[i] = base + i;
                    }
                }
                PUSH(((Value){.type=VAL_NIL})); /* return nil */
                break;
            }

            case 100: { /* build-string-from-packed: stack has [len, pack0, pack1, ...] */
                /* Read how many packs from the length */
                /* Stack layout: ..., len, pack0, pack1, ..., packN-1 (top) */
                /* We need to pop packs in reverse, then pop len */
                /* But we need len to know how many packs... */
                /* Convention: len is deepest, packs are on top */
                /* Calculate n_packs from the len value below the packs */
                /* This is tricky — we don't know sp offset for len without scanning. */
                /* Alternative: use the instruction's embedded data.
                 * Actually: NATIVE_CALL 100 is preceded by CONST(len), CONST(pack0), etc.
                 * The len value is at sp - n_packs - 1, packs at sp - n_packs .. sp - 1. */
                /* We'll read backwards: first pop all packs into a buffer, then pop len. */
                /* But we don't know n_packs without len! */
                /* Fix: pop values until we find the len marker.
                 * Len is always a small positive integer. Packs are large (multi-byte).
                 * This is fragile. Better approach: use a fixed protocol. */
                /* REVISED: the caller pushes len FIRST, then packs.
                 * At this point stack = [..., len, p0, p1, ..., pN-1]
                 * We need to peek at len to know N = (len+7)/8.
                 * Then pop N packs, pop len. */
                {
                    /* First, peek at the length value (it's below the packs) */
                    /* We need to find it. Count backwards from sp. */
                    /* Try: assume the len is the value at sp - n_packs - 1 where
                     * n_packs is determined by the len value itself. Chicken-and-egg.
                     * Solution: just try small values of n_packs. */
                    int slen = 0;
                    int npacks = 0;
                    /* Scan stack for the length value (must be < 256 and at position sp - k - 1
                     * where k = (len+7)/8) */
                    for (int try_len = 0; try_len < 256; try_len++) {
                        int try_np = (try_len + 7) / 8;
                        int len_pos = sp - try_np - 1;
                        if (len_pos >= 0 && stack[len_pos].type == VAL_INT && stack[len_pos].as.i == try_len) {
                            slen = try_len;
                            npacks = try_np;
                            break;
                        }
                    }
                    /* Pop packs */
                    int64_t packs[32];
                    for (int i = npacks - 1; i >= 0; i--) {
                        Value v = POP();
                        packs[i] = v.as.i;
                    }
                    POP(); /* pop len */
                    /* Build string */
                    int32_t sptr = HALLOC(); if (sptr < 0) break;
                    heap[sptr].type = HEAP_STRING;
                    heap[sptr].string.len = slen;
                    for (int i = 0; i < slen && i < 255; i++) {
                        int pack_idx = i / 8;
                        int byte_idx = i % 8;
                        heap[sptr].string.data[i] = (char)((packs[pack_idx] >> (byte_idx * 8)) & 0xFF);
                    }
                    heap[sptr].string.data[slen] = 0;
                    PUSH(((Value){.type=VAL_STRING,.as.ptr=sptr}));
                }
                break;
            }

            /* AD forward mode: dual numbers */
            case 110: { /* make-dual: value, derivative → dual number (stored as 2-element vector) */
                Value deriv = POP(), val = POP();
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = val;
                heap[ptr].vector.items[1] = deriv;
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 111: { /* dual-value: dual → value part */
                Value d = POP();
                if (d.type == VAL_VECTOR && heap[d.as.ptr].vector.len >= 2)
                    PUSH(heap[d.as.ptr].vector.items[0]);
                else PUSH(d); /* non-dual: value is itself */
                break;
            }
            case 112: { /* dual-derivative: dual → derivative part */
                Value d = POP();
                if (d.type == VAL_VECTOR && heap[d.as.ptr].vector.len >= 2)
                    PUSH(heap[d.as.ptr].vector.items[1]);
                else PUSH(FLOAT_VAL(0.0)); /* non-dual: derivative is 0 */
                break;
            }
            case 113: { /* dual-add: add two dual numbers */
                Value b = POP(), a = POP();
                double av, ad, bv, bd;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                if (b.type == VAL_VECTOR && heap[b.as.ptr].vector.len >= 2) {
                    bv = AS_NUM(heap[b.as.ptr].vector.items[0]);
                    bd = AS_NUM(heap[b.as.ptr].vector.items[1]);
                } else { bv = AS_NUM(b); bd = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(av + bv);
                heap[ptr].vector.items[1] = FLOAT_VAL(ad + bd);
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 114: { /* dual-mul: product rule (a,a')*(b,b') = (a*b, a*b'+a'*b) */
                Value b = POP(), a = POP();
                double av, ad, bv, bd;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                if (b.type == VAL_VECTOR && heap[b.as.ptr].vector.len >= 2) {
                    bv = AS_NUM(heap[b.as.ptr].vector.items[0]);
                    bd = AS_NUM(heap[b.as.ptr].vector.items[1]);
                } else { bv = AS_NUM(b); bd = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(av * bv);
                heap[ptr].vector.items[1] = FLOAT_VAL(av * bd + ad * bv);
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 115: { /* dual-sub: subtract dual numbers */
                Value b = POP(), a = POP();
                double av, ad, bv, bd;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                if (b.type == VAL_VECTOR && heap[b.as.ptr].vector.len >= 2) {
                    bv = AS_NUM(heap[b.as.ptr].vector.items[0]);
                    bd = AS_NUM(heap[b.as.ptr].vector.items[1]);
                } else { bv = AS_NUM(b); bd = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(av - bv);
                heap[ptr].vector.items[1] = FLOAT_VAL(ad - bd);
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 116: { /* dual-div: quotient rule (a,a')/(b,b') = (a/b, (a'*b-a*b')/b²) */
                Value b = POP(), a = POP();
                double av, ad, bv, bd;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                if (b.type == VAL_VECTOR && heap[b.as.ptr].vector.len >= 2) {
                    bv = AS_NUM(heap[b.as.ptr].vector.items[0]);
                    bd = AS_NUM(heap[b.as.ptr].vector.items[1]);
                } else { bv = AS_NUM(b); bd = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(av / bv);
                heap[ptr].vector.items[1] = FLOAT_VAL((ad * bv - av * bd) / (bv * bv));
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 117: { /* dual-sin: sin(a,a') = (sin(a), a'*cos(a)) */
                Value a = POP();
                double av, ad;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(sin(av));
                heap[ptr].vector.items[1] = FLOAT_VAL(ad * cos(av));
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 118: { /* dual-cos: cos(a,a') = (cos(a), -a'*sin(a)) */
                Value a = POP();
                double av, ad;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(cos(av));
                heap[ptr].vector.items[1] = FLOAT_VAL(-ad * sin(av));
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 119: { /* dual-exp: exp(a,a') = (exp(a), a'*exp(a)) */
                Value a = POP();
                double av, ad;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                double ea = exp(av);
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(ea);
                heap[ptr].vector.items[1] = FLOAT_VAL(ad * ea);
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 120: { /* dual-log: log(a,a') = (log(a), a'/a) */
                Value a = POP();
                double av, ad;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(log(av));
                heap[ptr].vector.items[1] = FLOAT_VAL(ad / av);
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }
            case 121: { /* dual-sqrt: sqrt(a,a') = (sqrt(a), a'/(2*sqrt(a))) */
                Value a = POP();
                double av, ad;
                if (a.type == VAL_VECTOR && heap[a.as.ptr].vector.len >= 2) {
                    av = AS_NUM(heap[a.as.ptr].vector.items[0]);
                    ad = AS_NUM(heap[a.as.ptr].vector.items[1]);
                } else { av = AS_NUM(a); ad = 0; }
                double sa = sqrt(av);
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = 2;
                heap[ptr].vector.items[0] = FLOAT_VAL(sa);
                heap[ptr].vector.items[1] = FLOAT_VAL(ad / (2.0 * sa));
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }

            case 260: { /* make-vector: TOS=fill, SOS=n → create vector of n elements filled with fill */
                Value fill = POP();
                Value n_val = POP();
                int n = (int)AS_NUM(n_val);
                if (n < 0) n = 0;
                if (n > 256) n = 256; /* safety limit */
                int32_t ptr = HALLOC(); if (ptr < 0) break;
                heap[ptr].type = HEAP_VECTOR;
                heap[ptr].vector.len = n;
                for (int i = 0; i < n; i++) {
                    heap[ptr].vector.items[i] = fill;
                }
                PUSH(((Value){.type=VAL_VECTOR,.as.ptr=ptr}));
                break;
            }

            default:
                fprintf(stderr, "ERROR: NATIVE_CALL %d not implemented\n", fid);
                error = 1;
                break;
            }
            break;
        }

        case OP_HALT: halted = 1; break;
        default: printf("  UNKNOWN OPCODE %d at PC=%d\n", ins.op, pc-1); error = 1; break;
        }
    }

    if (error) {
        fprintf(stderr, "=== VM ERROR at PC=%d ===\n", pc);
        fprintf(stderr, "  sp=%d fp=%d frame_count=%d heap=%d/%d\n",
                sp, fp, frame_count, heap_next, HEAP_SIZE);
        if (pc > 0 && pc <= chunk->code_len) {
            fprintf(stderr, "  last opcode: %d (operand=%d)\n",
                    chunk->code[pc-1].op, chunk->code[pc-1].operand);
        }
        fprintf(stderr, "  stack top 5:");
        for (int i = 0; i < 5 && i < sp; i++) {
            Value v = stack[sp - 1 - i];
            if (v.type == VAL_INT) fprintf(stderr, " %lld", (long long)v.as.i);
            else if (v.type == VAL_FLOAT) fprintf(stderr, " %.4g", v.as.f);
            else if (v.type == VAL_BOOL) fprintf(stderr, " %s", v.as.b ? "#t" : "#f");
            else if (v.type == VAL_NIL) fprintf(stderr, " nil");
            else fprintf(stderr, " <type=%d>", v.type);
        }
        fprintf(stderr, "\n");
    } else {
        printf("  [metrics] %lld insns, max_depth=%d, heap=%d/%d\n",
               (long long)insn_count, max_depth, heap_next, HEAP_SIZE);
    }

    #undef PUSH
    #undef POP
    #undef PEEK
    #undef AS_NUM
    #undef NUM_VAL
    #undef IS_FALSY
    #undef HALLOC

    free(stack); free(heap); free(frames);
}

/*******************************************************************************
 * Compile & Run
 ******************************************************************************/

/* Builtin function table: name → (native_id, arity) */
typedef struct { const char* name; int native_id; int arity; } BuiltinDef;

static const BuiltinDef BUILTINS[] = {
    /* Math (1 arg) */
    {"sin", 20, 1}, {"cos", 21, 1}, {"tan", 22, 1},
    {"exp", 23, 1}, {"log", 24, 1}, {"sqrt", 25, 1},
    {"floor", 26, 1}, {"ceiling", 27, 1}, {"round", 28, 1},
    {"asin", 29, 1}, {"acos", 30, 1}, {"atan", 31, 1},
    {"abs", 35, 1},  /* abs via native (not opcode) for first-class use */
    /* Math (2 arg) */
    {"expt", 32, 2}, {"min", 33, 2}, {"max", 34, 2},
    {"modulo", 36, 2}, {"remainder", 37, 2}, {"quotient", 38, 2},
    /* Predicates (1 arg) */
    {"positive?", 40, 1}, {"negative?", 41, 1},
    {"odd?", 42, 1}, {"even?", 43, 1},
    {"zero?", 44, 1},
    /* NOTE: null?, pair?, number?, boolean?, procedure?, vector?, car, cdr,
     * cons, display, list — these remain as compiler opcodes (not closures)
     * because they're core language primitives that must be visible at all scopes.
     * Only LIBRARY functions that need to be passed as arguments go here. */
    {"number->string", 51, 1},
    {"string-append", 54, 2}, {"string=?", 55, 2},
    {"string-length", 56, 1}, {"string-ref", 57, 2},
    {"newline", 60, 0},
    {"length", 71, 1},
    {"cadr", 77, 1}, {"cddr", 78, 1}, {"caar", 79, 1}, {"caddr", 80, 1},
    /* AD forward mode: dual number operations */
    {"make-dual", 110, 2},
    {"dual-value", 111, 1}, {"dual-derivative", 112, 1},
    {"dual+", 113, 2}, {"dual*", 114, 2}, {"dual-", 115, 2}, {"dual/", 116, 2},
    {"dual-sin", 117, 1}, {"dual-cos", 118, 1},
    {"dual-exp", 119, 1}, {"dual-log", 120, 1}, {"dual-sqrt", 121, 1},
    /* Equality */
    {"eq?", 133, 2}, {"eqv?", 133, 2}, {"equal?", 134, 2},
    /* List operations */
    {"append", 135, 2}, {"reverse", 136, 1},
    {"member", 137, 2}, {"assoc", 138, 2},
    {"list->vector", 139, 1}, {"vector->list", 140, 1},
    {"iota", 141, 1},
    /* Apply */
    {"apply", 70, 2},
    /* Arithmetic as first-class (2-arg, for use with apply/map/fold) */
    /* +,-,*,/ defined in scheme prelude as variadic folds */
    {"add2", 142, 2}, {"sub2", 143, 2}, {"mul2", 144, 2}, {"div2", 145, 2},
    /* Additional predicates */
    {"symbol?", 160, 1}, {"char?", 161, 1},
    {"exact?", 162, 1}, {"inexact?", 163, 1},
    {"nan?", 164, 1}, {"infinite?", 165, 1}, {"finite?", 166, 1},
    /* String operations */
    {"substring", 170, 3}, {"string-contains", 171, 2},
    {"string-upcase", 172, 1}, {"string-downcase", 173, 1},
    {"string-reverse", 174, 1},
    {"string->number", 175, 1}, {"number->string", 51, 1},
    {"string->list", 176, 1}, {"list->string", 177, 1},
    {"string-copy", 178, 1},
    /* Conversion */
    {"exact->inexact", 180, 1}, {"inexact->exact", 181, 1},
    {"char->integer", 182, 1}, {"integer->char", 183, 1},
    {"symbol->string", 184, 1}, {"string->symbol", 185, 1},
    /* Additional list */
    {"list-ref", 186, 2}, {"list-tail", 187, 2},
    {"last-pair", 188, 1}, {"list?", 189, 1},
    /* Math */
    {"truncate", 190, 1}, {"exact", 191, 1}, {"inexact", 192, 1},
    /* Hash tables */
    {"make-hash-table", 200, 0}, {"hash-ref", 201, 2}, {"hash-set!", 202, 3},
    {"hash-has-key?", 203, 2}, {"hash-keys", 204, 1}, {"hash-values", 205, 1},
    {"hash-count", 206, 1}, {"hash-delete!", 207, 2},
    /* Characters */
    {"char-alphabetic?", 210, 1}, {"char-numeric?", 211, 1},
    {"char-whitespace?", 212, 1}, {"char-upper-case?", 213, 1},
    {"char-lower-case?", 214, 1}, {"char-upcase", 215, 1},
    {"char-downcase", 216, 1}, {"char=?", 217, 2},
    {"char<?", 218, 2}, {"char>?", 219, 2},
    /* Bitwise */
    {"bitwise-and", 220, 2}, {"bitwise-or", 221, 2},
    {"bitwise-xor", 222, 2}, {"bitwise-not", 223, 1},
    {"arithmetic-shift", 224, 2},
    /* Additional list ops */
    {"take", 225, 2}, {"drop", 226, 2},
    {"any", 227, 2}, {"every", 228, 2},
    {"find", 229, 2}, {"sort", 230, 2},
    /* Additional string ops */
    {"string-repeat", 231, 2}, {"string-trim", 232, 1},
    {"string-split", 233, 2}, {"string-join", 234, 2},
    /* First-class core ops */
    {"cons", 74, 2}, {"car", 72, 1}, {"cdr", 73, 1},
    {"null?", 45, 1}, {"pair?", 46, 1}, {"number?", 47, 1},
    {"boolean?", 48, 1}, {"procedure?", 49, 1}, {"vector?", 50, 1},
    {"string?", 160, 1},
    /* Misc */
    {"not", 235, 1}, {"boolean=?", 236, 2},
    {"error", 237, 1}, {"void", 238, 0},
    {"hash-table?", 239, 1},
    {"display", 240, 1}, {"write", 241, 1},
    /* Complex numbers (300-319) */
    {"make-rectangular", 300, 2}, {"make-polar", 301, 2},
    {"real-part", 302, 1}, {"imag-part", 303, 1},
    {"magnitude", 304, 1}, {"angle", 305, 1},
    {"conjugate", 306, 1}, {"complex?", 317, 1},
    /* Rational numbers (330-349) */
    {"numerator", 331, 1}, {"denominator", 332, 1},
    {"exact->inexact", 343, 1}, {"inexact->exact", 344, 1},
    {"rationalize", 345, 2},
    /* AD — new-style IDs (370-399) */
    {"make-dual", 370, 2}, {"dual-primal", 371, 1}, {"dual-tangent", 372, 1},
    {"dual?", 383, 1},
    {"gradient", 750, 2}, {"jacobian", 751, 2}, {"hessian", 752, 2},
    {"derivative", 393, 2},
    /* Tensors (410-469) */
    {"make-tensor", 410, 2}, {"tensor-shape", 413, 1},
    {"tensor-reshape", 414, 2}, {"tensor-transpose", 415, 1},
    {"zeros", 417, 1}, {"ones", 418, 1},
    {"matmul", 440, 2}, {"softmax", 463, 1},
    /* Consciousness Engine (500-549) */
    {"logic-var?", 501, 1}, {"unify", 502, 3}, {"walk", 503, 2},
    {"make-substitution", 505, 0}, {"substitution?", 506, 1},
    {"make-fact", 507, 1}, {"fact?", 508, 1},
    {"make-kb", 509, 0}, {"kb?", 510, 1},
    {"kb-assert!", 511, 2}, {"kb-query", 512, 2},
    {"make-factor-graph", 520, 2}, {"factor-graph?", 521, 1},
    {"fg-add-factor!", 522, 3}, {"fg-infer!", 523, 3},
    {"free-energy", 525, 2}, {"expected-free-energy", 526, 3},
    {"make-workspace", 540, 2}, {"workspace?", 541, 1},
    {"ws-register!", 542, 3}, {"ws-step!", 543, 1},
    /* I/O (580-602) */
    {"open-input-file", 580, 1}, {"open-output-file", 581, 1},
    {"close-port", 582, 1}, {"read-char", 583, 1}, {"read-line", 585, 1},
    {"write-string", 587, 2}, {"eof-object?", 592, 1},
    {"open-input-string", 596, 1}, {"open-output-string", 597, 0},
    {"get-output-string", 598, 1}, {"file-exists?", 599, 1},
    /* Hash tables — new-style IDs (660-670) */
    {"make-hash-table", 660, 0}, {"hash-ref", 661, 3},
    {"hash-set!", 662, 3}, {"hash-has-key?", 663, 2},
    {"hash-remove!", 664, 2}, {"hash-keys", 665, 1},
    {"hash-values", 666, 1}, {"hash-count", 667, 1},
    {"hash-table?", 670, 1},
    /* Error objects (710-714) */
    {"error-object?", 711, 1}, {"error-object-message", 712, 1},
    {"error-object-irritants", 713, 1},
    /* Tensor ops (missing) */
    {"reshape", 414, 2}, {"tensor-get", 411, 2}, {"arange", 419, 1},
    /* Neural net ops (missing) */
    {"relu", 462, 1}, {"sigmoid", 464, 1}, {"conv2d", 465, 2}, {"dropout", 470, 2},
    {"mse-loss", 459, 2}, {"cross-entropy-loss", 460, 2},
    /* AD ops (missing) */
    {"divergence", 395, 2}, {"curl", 396, 2}, {"laplacian", 397, 2},
    /* Inference ops (missing) */
    {"fg-update-cpt!", 524, 3},
    /* Eshkol shorthands & missing builtins */
    {"vref", -1, 2},  /* uses OP_VEC_REF directly */
    {"diff", 393, 2}, {"tensor", 410, 2}, {"pow", 32, 2},
    {"type-of", 740, 1}, {"sign", 743, 1},
    /* Missing type predicates */
    {"real?", -1, 1}, {"rational?", 740, 1}, {"tensor?", 740, 1},
    {"port?", 730, 1}, {"input-port?", 728, 1}, {"output-port?", 729, 1},
    /* Missing math */
    {"cosh", 720, 1}, {"sinh", 721, 1}, {"tanh", 722, 1},
    /* Missing I/O */
    {"write-char", 586, 1}, {"write-line", 726, 1}, {"read", 588, 0},
    /* Missing tensor ops */
    {"tensor-ref", 411, 2}, {"tensor-sum", 445, 1}, {"tensor-mean", 446, 1},
    {"tensor-dot", 449, 2}, {"transpose", 415, 1}, {"flatten", 416, 1},
    {"linspace", 746, 3}, {"eye", 745, 1},
    {"model-save", 800, 2}, {"model-load", 801, 1},
    {"tensor-save", 802, 2}, {"tensor-load", 803, 1},
    /* Standalone VM system surfaces */
    {"process-spawn", 1780, 3}, {"process-wait", 1781, 1},
    {"process-spawn-with-env", 1780, 3},
    {"process-kill", 1782, 2}, {"io-poll", 1783, 2},
    {"poll", 1783, 2}, {"process-pid", 1784, 0},
    {"process-setpgid", 1785, 2}, {"process-kill-tree", 1786, 2},
    {"process-spawn-pty", 1787, 1}, {"process-read-nonblocking", 1788, 2},
    {"fork", 1789, 0},
    {"unix-socket-connect", 1790, 1}, {"socket-send", 1791, 2},
    {"socket-recv", 1792, 2}, {"socket-close", 1793, 1},
    {"signal-install", 1794, 1}, {"signal-check", 1795, 0},
    {"signal-reset", 1796, 1}, {"signal-ignore", 1797, 1},
    {"signal-count", 1798, 0},
    {"execv", 1799, 2},
    {"make-temp-file", 1760, 3}, {"make-temp-dir", 1761, 2},
    {"term-set-scroll-region", 1930, 2}, {"term-reset-scroll-region", 1931, 0},
    {"term-enable-mouse", 1932, 0}, {"term-disable-mouse", 1933, 0},
    {"term-read-mouse-event", 1934, 1},
    {"term-enable-alternate-screen", 1935, 0},
    {"term-disable-alternate-screen", 1936, 0},
    {"term-clipboard-write", 1937, 1}, {"term-clipboard-read", 1938, 0},
    {"term-hyperlink", 1939, 2}, {"term-detect-capabilities", 1940, 0},
    {"term-bell", 1941, 0},
    {"fs-watch-native", 1942, 2}, {"fs-watch-recursive", 1943, 2},
    {"fs-watch-poll", 1944, 1}, {"fs-unwatch", 1945, 1},
    {"ansi-strip", 1946, 1}, {"string-display-width", 1947, 1},
    {"string-truncate-display", 1948, 3},
    {"executable-path", 1949, 1}, {"monotonic-time-ms", 1950, 0},
    {"temp-directory", 1951, 0}, {"prevent-sleep", 1952, 1},
    {"allow-sleep", 1953, 1},
    {"url-encode", 1954, 1}, {"url-decode", 1955, 1},
    {"url-parse", 1960, 1},
    {"base64url-encode", 1961, 1}, {"base64url-decode", 1962, 1},
    {"uuid-v4", 1963, 0}, {"constant-time-equal?", 1964, 2},
    {"sha256-file", 1965, 1},
    {"regex-compile", 1966, 1}, {"regex-free", 1967, 1},
    {"regex-match", 1968, 2}, {"regex-match?", 1969, 2},
    {"regex-match-groups", 1970, 2}, {"regex-split", 1971, 2},
    {"current-timestamp", 1972, 0}, {"current-time-ns", 1973, 0},
    {"format-iso8601", 1974, 1}, {"parse-iso8601", 1975, 1},
    {"format-relative", 1976, 1}, {"local-timezone-offset", 1977, 0},
    {"diff-lines", 1978, 2}, {"fuzzy-match", 1979, 4},
    {"semver-parse", 1980, 1}, {"semver-compare", 1981, 2},
    {"semver-satisfies?", 1982, 2},
    {"make-pipe", 1983, 0}, {"fd-write", 1984, 2},
    {"make-line-reader", 1985, 2}, {"line-reader-poll", 1986, 1},
    {"line-reader-close", 1987, 1}, {"fd-close", 1988, 1},
    {"make-lru-cache", 1989, 1}, {"lru-get", 1990, 2},
    {"lru-set!", 1991, 3}, {"lru-has?", 1992, 2},
    {"lru-delete!", 1993, 2}, {"lru-clear!", 1994, 1},
    {"lru-size", 1995, 1},
    {"_emit-event", 1996, 3}, {"make-event-emitter", 1997, 0},
    {"on!", 1998, 3}, {"once!", 1999, 3}, {"off!", 2000, 3},
    {"make-channel", 2001, 1}, {"channel-send!", 2002, 2},
    {"channel-receive", 2003, 2}, {"channel-recv!", 2003, 2},
    {"channel-try-receive", 2004, 1}, {"channel-try-recv!", 2004, 1},
    {"channel-close!", 2005, 1}, {"make-mutex", 2006, 0},
    {"mutex-lock!", 2007, 1}, {"mutex-unlock!", 2008, 1},
    {"with-mutex", 2009, 2}, {"make-condition-variable", 2010, 0},
    {"make-condvar", 2010, 0}, {"condition-wait", 2011, 2},
    {"condvar-wait!", 2011, 2}, {"condition-signal", 2012, 1},
    {"condvar-signal!", 2012, 1}, {"condition-broadcast", 2013, 1},
    {"condvar-broadcast!", 2013, 1},
    {"json-get-in", 2014, 3}, {"json-stringify-pretty", 2015, 2},
    {"json-merge", 2016, 2},
    {"compression-available", 2017, 0}, {"deflate", 2018, 1},
    {"inflate", 2019, 1}, {"gzip", 2020, 1}, {"gunzip", 2021, 1},
    {"make-timer", 2022, 2}, {"timer-cancel!", 2023, 1},
    {"make-interval", 2024, 2}, {"interval-cancel!", 2025, 1},
    {"timer-check", 2026, 1},
    {"db-transaction", 2027, 2}, {"db-busy-timeout", 2028, 2},
    {"db-last-insert-id", 2029, 1}, {"db-changes", 2030, 1},
    {"at-exit", 2031, 1},
    {"dlopen", 2032, 1}, {"dlsym", 2033, 2}, {"dlclose", 2034, 1},
    {"_format-list", 2035, 2},
    {"yoga-node-create", 2036, 0}, {"yoga-node-set!", 2037, 3},
    {"yoga-node-add-child!", 2038, 2}, {"yoga-node-calculate!", 2039, 3},
    {"yoga-node-get-computed", 2040, 2}, {"yoga-node-free!", 2041, 1},
    {"http-server-create", 2042, 1}, {"http-server-port", 2043, 1},
    {"http-server-accept", 2044, 3}, {"http-server-respond", 2045, 4},
    {"http-server-close", 2046, 1},
    {"http-request", 2047, 5},
    {"websocket-connect", 2048, 2}, {"websocket-send", 2049, 2},
    {"websocket-send-binary", 2050, 2}, {"websocket-receive", 2051, 2},
    {"websocket-close", 2052, 1},
    {"ts-parser-new", 2053, 1}, {"ts-parser-free", 2054, 1},
    {"ts-parse", 2055, 2}, {"ts-tree-free", 2056, 1},
    {"ts-node-type", 2057, 1}, {"ts-node-text", 2058, 2},
    {"ts-node-children", 2059, 1}, {"ts-query-new", 2060, 2},
    {"ts-query-matches", 2061, 3}, {"ts-query-free", 2062, 1},
    {"ts-available", 2063, 0}, {"ts-tree-root", 2064, 1},
    {"http-set-proxy", 2065, 1}, {"http-set-tls-client-cert", 2066, 3},
    {"display-error", 2067, 1},
    {"open-binary-input-file", 2068, 1}, {"open-binary-output-file", 2069, 1},
    {"read-u8", 2070, 1}, {"write-u8", 2071, 2},
    {"read-bytevector", 2072, 2}, {"write-bytevector", 2073, 2},
    {"setenv", 1712, 2}, {"unsetenv", 1713, 1},
    {"getenv", 1715, 1}, {"get-environment-variable", 1715, 1},
    {"string-ends-with?", 1956, 2}, {"string-index-of", 1957, 3},
    {"string-pad-left", 1958, 3}, {"string-pad-right", 1959, 3},
    /* Missing hash */
    {"hash-clear!", 668, 1},
    /* gcd / lcm */
    {"gcd", 346, 2}, {"lcm", 347, 2},
    {NULL, 0, 0}  /* sentinel */
};

/**
 * @brief Emit the preamble that defines every entry in the BUILTINS table
 *        as a first-class closure local (a small JUMP-over-body/GET_LOCAL.../
 *        NATIVE_CALL/RETURN/CLOSURE sequence per builtin), so builtins can
 *        be passed as ordinary values/arguments (e.g. `(map even? lst)`).
 */
