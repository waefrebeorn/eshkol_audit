static void chunk_emit(FuncChunk* c, uint8_t op, int32_t operand) {
    if (c->code_len >= MAX_CODE) { fprintf(stderr, "ERROR: bytecode overflow (MAX_CODE=%d)\n", MAX_CODE); return; }
    c->code[c->code_len++] = (Instr){op, operand};
}

/** @brief Append a value to chunk @p c's constant pool. Deliberately does
 *         not deduplicate: function PC placeholders get patched in place
 *         after creation, which would corrupt any literal constant that
 *         happened to match the placeholder's value.
 * @return The new constant's index, or -1 if MAX_CONSTS is exceeded.
 */
static int chunk_add_const(FuncChunk* c, Value v) {
    /* No deduplication — function PC placeholders get patched after creation,
     * which would corrupt literal constants that matched the placeholder value. */
    if (c->n_constants >= MAX_CONSTS) { fprintf(stderr, "ERROR: constant pool overflow (MAX_CONSTS=%d)\n", MAX_CONSTS); return -1; }
    c->constants[c->n_constants] = v;
    return c->n_constants++;
}

/** @brief Emit an OP_NOP placeholder instruction (e.g. for a forward jump
 *         target to be patch()ed once its destination is known).
 * @return The code index of the placeholder slot.
 */
static int placeholder(FuncChunk* c) {
    int slot = c->code_len;
    chunk_emit(c, OP_NOP, 0);
    return slot;
}

/** @brief Overwrite the instruction at @p slot (previously emitted by
 *         placeholder() or otherwise) with (@p op, @p target). */
static void patch(FuncChunk* c, int slot, uint8_t op, int32_t target) {
    c->code[slot] = (Instr){op, target};
}

/** @brief Look up a local variable by name in the innermost-to-outermost
 *         scope of chunk @p c.
 * @return The local's stack slot, or -1 if not found in this chunk.
 */
static int resolve_local(FuncChunk* c, const char* name) {
    for (int i = c->n_locals - 1; i >= 0; i--) {
        if (strcmp(c->locals[i].name, name) == 0) return c->locals[i].slot;
    }
    return -1;
}

/** @brief Register a new local variable named @p name in chunk @p c at the
 *         current scope depth.
 * @return The assigned slot index, or -1 if MAX_LOCALS is exceeded.
 */
static int add_local(FuncChunk* c, const char* name) {
    if (c->n_locals >= MAX_LOCALS) { fprintf(stderr, "ERROR: local variable overflow (MAX_LOCALS=%d)\n", MAX_LOCALS); return -1; }
    int slot = c->n_locals;
    strncpy(c->locals[c->n_locals].name, name, 127);
    c->locals[c->n_locals].name[127] = 0;
    c->locals[c->n_locals].slot = slot;
    c->locals[c->n_locals].depth = c->scope_depth;
    c->n_locals++;
    return slot;
}

static void compile_expr(FuncChunk* c, Node* node, int tail_position);

/** @brief Scan an AST subtree for a `(set! name ...)` reference to
 *         @p name. */
static int scan_for_set(Node* node, const char* name) {
    if (!node) return 0;
    if (node->type == N_LIST && node->n_children >= 3) {
        Node* head = node->children[0];
        if (head->type == N_SYMBOL && strcmp(head->symbol, "set!") == 0
            && node->children[1]->type == N_SYMBOL
            && strcmp(node->children[1]->symbol, name) == 0)
            return 1;
    }
    if (node->type == N_LIST) {
        for (int i = 0; i < node->n_children; i++)
            if (scan_for_set(node->children[i], name)) return 1;
    }
    return 0;
}

/** @brief Scan for free references to @p name inside nested lambda bodies —
 *         a reference is free (a capture) if @p name is not rebound as a
 *         lambda parameter or let binding at an inner scope.
 *         @p in_lambda tracks whether the current recursion is inside a
 *         lambda body. */
static int scan_for_capture(Node* node, const char* name, int in_lambda) {
    if (!node) return 0;
    if (node->type == N_SYMBOL && in_lambda && strcmp(node->symbol, name) == 0)
        return 1;
    if (node->type == N_LIST && node->n_children >= 1) {
        Node* head = node->children[0];
        /* Check if this lambda/let rebinds the variable — if so, it's not a capture */
        if (head->type == N_SYMBOL && strcmp(head->symbol, "lambda") == 0 && node->n_children >= 3) {
            /* Check if name is a parameter of this lambda */
            Node* params = node->children[1];
            if (params->type == N_LIST) {
                for (int i = 0; i < params->n_children; i++)
                    if (params->children[i]->type == N_SYMBOL && strcmp(params->children[i]->symbol, name) == 0)
                        return 0; /* rebound as parameter — not a capture */
            }
            /* Scan body (now inside lambda) */
            for (int i = 2; i < node->n_children; i++)
                if (scan_for_capture(node->children[i], name, 1)) return 1;
            return 0;
        }
        if (head->type == N_SYMBOL && (strcmp(head->symbol, "let") == 0 ||
            strcmp(head->symbol, "let*") == 0 || strcmp(head->symbol, "letrec") == 0)) {
            /* Check if name is rebound in this let's bindings */
            if (node->n_children >= 3 && node->children[1]->type == N_LIST) {
                Node* bindings = node->children[1];
                for (int i = 0; i < bindings->n_children; i++) {
                    Node* b = bindings->children[i];
                    if (b->type == N_LIST && b->n_children >= 1 && b->children[0]->type == N_SYMBOL
                        && strcmp(b->children[0]->symbol, name) == 0)
                        return 0; /* rebound in inner let */
                }
            }
        }
        /* Recurse into children */
        int new_lambda = in_lambda;
        if (head->type == N_SYMBOL && strcmp(head->symbol, "lambda") == 0)
            new_lambda = 1;
        for (int i = 0; i < node->n_children; i++)
            if (scan_for_capture(node->children[i], name, new_lambda)) return 1;
    }
    return 0;
}

/** @brief Check whether a let-bound variable @p name needs heap boxing:
 *         true only if it is both `set!`-mutated (scan_for_set()) and
 *         captured by a nested lambda (scan_for_capture()) somewhere across
 *         @p body_nodes. */
static int needs_boxing(Node* body_nodes[], int n_bodies, const char* name) {
    int has_set = 0, has_capture = 0;
    for (int i = 0; i < n_bodies; i++) {
        if (scan_for_set(body_nodes[i], name)) has_set = 1;
        if (scan_for_capture(body_nodes[i], name, 0)) has_capture = 1;
    }
    return has_set && has_capture;
}

/** @brief Compile a `(quote datum)` literal: numbers/booleans/strings as
 *         constants, symbols as packed 8-byte constant chunks passed to
 *         native call 100 (symbol construction), and lists as a chain of
 *         OP_CONS built from an OP_NIL base (right to left). */
static void compile_quote(FuncChunk* c, Node* datum) {
    if (!datum) { chunk_emit(c, OP_NIL, 0); return; }
    if (datum->type == N_NUMBER) {
        double v = datum->numval;
        if (v == (int64_t)v && fabs(v) < 1e15)
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL((int64_t)v)));
        else
            chunk_emit(c, OP_CONST, chunk_add_const(c, FLOAT_VAL(v)));
        return;
    }
    if (datum->type == N_BOOL) {
        chunk_emit(c, datum->numval ? OP_TRUE : OP_FALSE, 0);
        return;
    }
    if (datum->type == N_STRING) {
        compile_expr(c, datum, 0); /* reuse string literal compilation */
        return;
    }
    if (datum->type == N_SYMBOL) {
        /* Quoted symbol → compile as string */
        int len = (int)strlen(datum->symbol);
        int n_packs = (len + 7) / 8;
        chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(len)));
        for (int p = 0; p < n_packs; p++) {
            int64_t pack = 0;
            for (int b = 0; b < 8 && p * 8 + b < len; b++)
                pack |= ((int64_t)(unsigned char)datum->symbol[p * 8 + b]) << (b * 8);
            chunk_emit(c, OP_CONST, chunk_add_const(c, INT_VAL(pack)));
        }
        chunk_emit(c, OP_NATIVE_CALL, 100);
        return;
    }
    if (datum->type == N_LIST) {
        if (datum->n_children == 0) { chunk_emit(c, OP_NIL, 0); return; }
        /* Build proper list: (cons el0 (cons el1 ... (cons elN-1 '()))) */
        /* Compile in reverse: push NIL, then cons each element from back to front */
        chunk_emit(c, OP_NIL, 0);
        for (int i = datum->n_children - 1; i >= 0; i--) {
            compile_quote(c, datum->children[i]);
            chunk_emit(c, OP_CONS, 0);
        }
        return;
    }
    chunk_emit(c, OP_NIL, 0); /* fallback */
}

