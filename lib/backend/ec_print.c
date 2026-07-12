static void print_value(Value v, HeapObject* heap, int depth, int mode);

/**
 * @brief Recursively print a runtime Value (bounded to @p depth <= 50 to
 *        avoid infinite recursion on circular structures — prints "..."
 *        past that). @p mode 0 = display (raw, human-readable strings), 1
 *        = write (quoted/escaped strings, machine-readable).
 */
static void print_value(Value v, HeapObject* heap, int depth, int mode) {
    if (depth > 50) { printf("..."); return; }
    switch (v.type) {
    case VAL_INT:   printf("%lld", (long long)v.as.i); break;
    case VAL_FLOAT: printf("%.6g", v.as.f); break;
    case VAL_BOOL:  printf("%s", v.as.b ? "#t" : "#f"); break;
    case VAL_NIL:   printf("()"); break;
    case VAL_CLOSURE: printf("<closure>"); break;
    case VAL_CONTINUATION: printf("<continuation>"); break;
    case VAL_HASH: printf("<hash-table:%d>", heap[v.as.ptr].hash.count); break;
    case VAL_STRING:
        if (mode == 1) {
            /* write mode: output with quotes and escape sequences */
            printf("\"");
            const char* s = heap[v.as.ptr].string.data;
            int len = heap[v.as.ptr].string.len;
            for (int i = 0; i < len; i++) {
                switch (s[i]) {
                case '"':  printf("\\\""); break;
                case '\\': printf("\\\\"); break;
                case '\n': printf("\\n"); break;
                case '\t': printf("\\t"); break;
                case '\r': printf("\\r"); break;
                default:   putchar(s[i]); break;
                }
            }
            printf("\"");
        } else {
            /* display mode: output raw string contents */
            printf("%.*s", heap[v.as.ptr].string.len, heap[v.as.ptr].string.data);
        }
        break;
    case VAL_PAIR: {
        printf("(");
        Value cur = v; int first = 1;
        while (cur.type == VAL_PAIR && depth < 50) {
            if (!first) printf(" "); first = 0;
            print_value(heap[cur.as.ptr].cons.car, heap, depth + 1, mode);
            cur = heap[cur.as.ptr].cons.cdr;
        }
        if (cur.type != VAL_NIL) { printf(" . "); print_value(cur, heap, depth + 1, mode); }
        printf(")");
        break;
    }
    case VAL_VECTOR: {
        printf("#(");
        for (int i = 0; i < heap[v.as.ptr].vector.len; i++) {
            if (i > 0) printf(" ");
            print_value(heap[v.as.ptr].vector.items[i], heap, depth + 1, mode);
        }
        printf(")");
        break;
    }
    default: printf("<unknown>"); break;
    }
}

/**
 * @brief Run a fixed-point pass of local bytecode peephole optimizations
 *        over chunk @p c: eliminates `CONST 0 + ADD` and `CONST 1 + MUL`
 *        identities, `NOT + NOT` / `NEG + NEG` double-negation pairs, and
 *        `DUP + POP` pairs, replacing eliminated instructions with OP_NOP
 *        in place (never compacted, since compaction would require fixing
 *        up jump targets — the VM treats NOP as near-zero cost). Prints a
 *        summary of eliminated instruction count.
 */
