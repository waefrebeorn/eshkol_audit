static void emit_builtin_preamble(FuncChunk* c) {
    for (int b = 0; BUILTINS[b].name; b++) {
        const BuiltinDef* def = &BUILTINS[b];
        int func_slot = add_local(c, def->name);

        /* Emit: JUMP over body → body (GETL params, NATIVE_CALL, RET) → CLOSURE */
        int cfunc = chunk_add_const(c, INT_VAL(0)); /* placeholder for func PC */
        int jover = placeholder(c);

        int func_pc = c->code_len;
        c->constants[cfunc].as.i = func_pc;

        /* Function body: load args from local slots, call native, return */
        for (int a = 0; a < def->arity; a++) {
            chunk_emit(c, OP_GET_LOCAL, a);
        }
        chunk_emit(c, OP_NATIVE_CALL, def->native_id);
        chunk_emit(c, OP_RETURN, 0);

        patch(c, jover, OP_JUMP, c->code_len);
        chunk_emit(c, OP_CLOSURE, cfunc); /* 0 upvalues */
        /* Closure is now on stack at func_slot */
    }
}

/* Global ESKB output path (set by --emit-eskb flag in main) */
static const char* g_eskb_output_path = NULL;
static const char* g_source_file_path = NULL;

/**
 * @brief Top-level driver: builds the main FuncChunk by emitting the
 *        builtin preamble (emit_builtin_preamble()), compiling a
 *        Scheme-level prelude (map/filter/fold-left/fold-right/for-each/
 *        any/every/find/take/drop/etc., implemented in terms of the
 *        builtin closures), then parsing and compiling @p source itself.
 *        Runs peephole_optimize() on the result and either executes it
 *        (execute_chunk()) or writes it to g_eskb_output_path in ESKB
 *        binary format, depending on the flags set by main().
 */
static void compile_and_run(const char* source) {
    FuncChunk main_chunk = {0};

    /* Emit builtin function definitions as first-class closures */
    emit_builtin_preamble(&main_chunk);
    /* stack_depth synced via n_locals */

    /* Compile Scheme-level builtins (higher-order functions that call closures) */
    static const char* scheme_prelude =
        "(define (map f lst)\n"
        "  (let loop ((l lst) (acc (list)))\n"
        "    (if (null? l) (reverse acc)\n"
        "      (loop (cdr l) (cons (f (car l)) acc)))))\n"
        "(define (filter pred lst)\n"
        "  (let loop ((l lst) (acc (list)))\n"
        "    (if (null? l) (reverse acc)\n"
        "      (if (pred (car l)) (loop (cdr l) (cons (car l) acc))\n"
        "        (loop (cdr l) acc)))))\n"
        "(define (fold-left f init lst)\n"
        "  (let loop ((l lst) (acc init))\n"
        "    (if (null? l) acc\n"
        "      (loop (cdr l) (f acc (car l))))))\n"
        "(define (fold-right f init lst) (if (null? lst) init (f (car lst) (fold-right f init (cdr lst)))))\n"
        "(define (for-each f lst) (if (null? lst) 0 (begin (f (car lst)) (for-each f (cdr lst)))))\n"
        "(define (any pred lst) (if (null? lst) #f (if (pred (car lst)) #t (any pred (cdr lst)))))\n"
        "(define (every pred lst) (if (null? lst) #t (if (pred (car lst)) (every pred (cdr lst)) #f)))\n"
        "(define (find pred lst) (if (null? lst) #f (if (pred (car lst)) (car lst) (find pred (cdr lst)))))\n"
        "(define (take n lst) (if (= n 0) (list) (if (null? lst) (list) (cons (car lst) (take (- n 1) (cdr lst))))))\n"
        "(define (drop n lst) (if (= n 0) lst (if (null? lst) (list) (drop (- n 1) (cdr lst)))))\n"
        "(define (reduce f init lst) (fold-left f init lst))\n"
        "(define (merge compare a b)\n"
        "  (cond ((null? a) b) ((null? b) a)\n"
        "    ((compare (car a) (car b)) (cons (car a) (merge compare (cdr a) (cdr b))))\n"
        "    (else (cons (car b) (merge compare a (cdr b))))))\n"
        "(define (sort compare lst)\n"
        "  (if (or (null? lst) (null? (cdr lst))) lst\n"
        "    (let ((half (quotient (length lst) 2)))\n"
        "      (merge compare (sort compare (take half lst)) (sort compare (drop half lst))))))\n"
        "(define + (lambda args (fold-left add2 0 args)))\n"
        "(define * (lambda args (fold-left mul2 1 args)))\n"
        "(define (- . args) (if (null? (cdr args)) (sub2 0 (car args)) (fold-left sub2 (car args) (cdr args))))\n"
        "(define (/ . args) (if (null? (cdr args)) (div2 1 (car args)) (fold-left div2 (car args) (cdr args))))\n"
        "(define (format fmt . args) (_format-list fmt args))\n"
        "(define (emit! emitter event . args) (_emit-event emitter event args))\n";
    src_ptr = scheme_prelude;
    while (1) {
        skip_ws();
        if (!*src_ptr) break;
        Node* expr = parse_sexp();
        if (!expr) break;
        int locals_before = main_chunk.n_locals;
        compile_expr(&main_chunk, expr, 0);
        if (main_chunk.n_locals == locals_before)
            chunk_emit(&main_chunk, OP_POP, 0);
        free_node(expr);
    }

    /* stack_depth synced via n_locals */
    src_ptr = source;

    /* TWO-PASS COMPILATION:
     * Pass 1: Parse ALL top-level expressions into an AST array.
     * Pass 2: Scan for defines that need heap boxing (captured + mutated).
     * Pass 3: Compile with boxing information. */

    /* Pass 1: Parse */
    #define MAX_TOP_EXPRS 4096
    Node* top_exprs[MAX_TOP_EXPRS];
    int n_top_exprs = 0;
    while (1) {
        skip_ws();
        if (!*src_ptr) break;
        Node* expr = parse_sexp();
        if (!expr) break;
        if (n_top_exprs < MAX_TOP_EXPRS)
            top_exprs[n_top_exprs++] = expr;
    }

    /* Pass 2: Scan for top-level defines that need boxing.
     * A define needs boxing if its variable is both:
     * (a) captured by a lambda somewhere in the program, AND
     * (b) mutated via set! somewhere in the program.
     * We record which define names need boxing. */
    char boxed_names[256][128];
    int n_boxed = 0;
    for (int i = 0; i < n_top_exprs; i++) {
        Node* expr = top_exprs[i];
        /* Check if this is a simple define: (define name value) */
        if (expr->type == N_LIST && expr->n_children >= 3
            && expr->children[0]->type == N_SYMBOL
            && strcmp(expr->children[0]->symbol, "define") == 0
            && expr->children[1]->type == N_SYMBOL) {
            const char* name = expr->children[1]->symbol;
            /* Scan ALL subsequent expressions for set! + capture */
            int has_set = 0, has_capture = 0;
            for (int j = 0; j < n_top_exprs; j++) {
                if (scan_for_set(top_exprs[j], name)) has_set = 1;
                if (scan_for_capture(top_exprs[j], name, 0)) has_capture = 1;
            }
            if (has_set && has_capture && n_boxed < 256) {
                strncpy(boxed_names[n_boxed], name, 127);
                boxed_names[n_boxed][127] = 0;
                n_boxed++;
            }
        }
    }

    /* Pass 3: Compile with boxing */
    for (int i = 0; i < n_top_exprs; i++) {
        Node* expr = top_exprs[i];

        /* Check if this is a simple define that needs boxing */
        int do_box = 0;
        if (expr->type == N_LIST && expr->n_children >= 3
            && expr->children[0]->type == N_SYMBOL
            && strcmp(expr->children[0]->symbol, "define") == 0
            && expr->children[1]->type == N_SYMBOL) {
            const char* name = expr->children[1]->symbol;
            for (int b = 0; b < n_boxed; b++) {
                if (strcmp(boxed_names[b], name) == 0) { do_box = 1; break; }
            }
        }

        int locals_before = main_chunk.n_locals;

        if (do_box) {
            /* Compile the init value, wrap in a box (1-element vector) */
            compile_expr(&main_chunk, expr->children[2], 0);
            chunk_emit(&main_chunk, OP_VEC_CREATE, 1); /* box it */
            int slot = add_local(&main_chunk, expr->children[1]->symbol);
            main_chunk.locals[main_chunk.n_locals - 1].boxed = 1;
        } else {
            compile_expr(&main_chunk, expr, 0);
            if (main_chunk.n_locals == locals_before) {
                chunk_emit(&main_chunk, OP_POP, 0);
            }
        }
    }

    /* Free ASTs */
    for (int i = 0; i < n_top_exprs; i++)
        free_node(top_exprs[i]);
    chunk_emit(&main_chunk, OP_HALT, 0);

    /* Print bytecode summary */
    printf("  [compiled: %d instructions, %d constants, %d locals]\n",
           main_chunk.code_len, main_chunk.n_constants, main_chunk.n_locals);

    /* Disassemble */
    static const char* opn[] = {
        "NOP","CONST","NIL","TRUE","FALSE","POP","DUP",
        "ADD","SUB","MUL","DIV","MOD","NEG","ABS",
        "EQ","LT","GT","LE","GE","NOT",
        "GETL","SETL","GETUP","SETUP",
        "CLOS","CALL","TCALL","RET",
        "JUMP","JIF","LOOP",
        "CONS","CAR","CDR","NULLP",
        "PRINT","HALT","NATV","CLOSUP",
        "VECNW","VECRF","VECST","VECLN",
        "STRRF","STRLN",
        "PAIRP","NUMP","STRP","BOOLP","PROCP","VECP",
        "SETCR","SETCD","POPN","OCLOS","CCALL","IVCC",
        "GUARD","UNGRD","GETXN","PKRST","WNDPS","WNDPP"
    };
    for (int i = 0; i < main_chunk.code_len; i++) {
        Instr ins = main_chunk.code[i];
        printf("    [%3d] %-6s %d", i, ins.op < OP_COUNT ? opn[ins.op] : "???", ins.operand);
        if (ins.op == OP_CONST && ins.operand < main_chunk.n_constants) {
            Value v = main_chunk.constants[ins.operand];
            if (v.type == VAL_INT) printf("  ; %lld", (long long)v.as.i);
        }
        if (ins.op == OP_CLOSURE) printf("  ; func@%lld, %d upvals",
            (long long)main_chunk.constants[ins.operand & 0xFFFF].as.i,
            (ins.operand >> 16) & 0xFF);
        printf("\n");
    }

    /* Dump bytecode for weight matrix integration (if requested) */
    if (getenv("ESHKOL_DUMP_BC")) {
        const char* path = getenv("ESHKOL_DUMP_BC");
        FILE* bf = fopen(path, "wb");
        if (bf) {
            uint32_t magic = 0x45534B42; /* "ESKB" */
            uint32_t n_instr = main_chunk.code_len;
            uint32_t n_const = main_chunk.n_constants;
            fwrite(&magic, 4, 1, bf);
            fwrite(&n_instr, 4, 1, bf);
            fwrite(&n_const, 4, 1, bf);
            /* Write instructions as (op:u8, operand:i32) pairs */
            for (int i = 0; i < (int)n_instr; i++) {
                uint8_t op = main_chunk.code[i].op;
                int32_t operand = main_chunk.code[i].operand;
                fwrite(&op, 1, 1, bf);
                fwrite(&operand, 4, 1, bf);
            }
            /* Write constants as (type:u8, value:f64) pairs */
            for (int i = 0; i < (int)n_const; i++) {
                uint8_t type = main_chunk.constants[i].type;
                double val = 0;
                if (type == VAL_INT) val = (double)main_chunk.constants[i].as.i;
                else if (type == VAL_FLOAT) val = main_chunk.constants[i].as.f;
                else if (type == VAL_BOOL) val = (double)main_chunk.constants[i].as.b;
                fwrite(&type, 1, 1, bf);
                fwrite(&val, 8, 1, bf);
            }
            fclose(bf);
            printf("  [dumped bytecode: %d instructions, %d constants → %s]\n",
                   (int)n_instr, (int)n_const, path);
        }
    }

    /* Emit ESKB binary format (if --emit-eskb was requested via global) */
    if (g_eskb_output_path) {
        /* Convert FuncChunk constants and code to ESKB format */
        EskbInstr* eskb_code = (EskbInstr*)calloc(main_chunk.code_len, sizeof(EskbInstr));
        EskbConst* eskb_consts = (EskbConst*)calloc(main_chunk.n_constants > 0 ? main_chunk.n_constants : 1, sizeof(EskbConst));
        if (eskb_code && eskb_consts) {
            for (int i = 0; i < main_chunk.code_len; i++) {
                eskb_code[i].op = main_chunk.code[i].op;
                eskb_code[i].operand = main_chunk.code[i].operand;
            }
            for (int i = 0; i < main_chunk.n_constants; i++) {
                Value v = main_chunk.constants[i];
                switch (v.type) {
                case VAL_NIL:
                    eskb_consts[i].type = ESKB_CONST_NIL;
                    break;
                case VAL_INT:
                    eskb_consts[i].type = ESKB_CONST_INT64;
                    eskb_consts[i].as.i = v.as.i;
                    break;
                case VAL_FLOAT:
                    eskb_consts[i].type = ESKB_CONST_F64;
                    eskb_consts[i].as.f = v.as.f;
                    break;
                case VAL_BOOL:
                    eskb_consts[i].type = ESKB_CONST_BOOL;
                    eskb_consts[i].as.b = v.as.b;
                    break;
                default:
                    /* Closures, pairs, etc. — store as int64 */
                    eskb_consts[i].type = ESKB_CONST_INT64;
                    eskb_consts[i].as.i = v.as.i;
                    break;
                }
            }
            eskb_write_file(g_eskb_output_path, eskb_code, main_chunk.code_len,
                            eskb_consts, main_chunk.n_constants, g_source_file_path);
        }
        free(eskb_code);
        free(eskb_consts);
    }

    /* Run peephole optimization before execution */
    peephole_optimize(&main_chunk);

    /* Execute using full VM */
    execute_chunk(&main_chunk);
}

/**
 * @brief CLI entry point: parses `--emit-eskb PATH` (write compiled
 *        bytecode to an .eskb binary instead of / in addition to running
 *        it) and `--trace` flags, locates the input source file argument,
 *        reads it, and calls compile_and_run().
 */
int main(int argc, char** argv) {
    printf("=== Eshkol Compiler (targeting 38-opcode VM) ===\n\n");

    /* Check for --emit-eskb and --trace flags */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--emit-eskb") == 0 && i + 1 < argc) {
            g_eskb_output_path = argv[++i];
        }
        if (strcmp(argv[i], "--trace") == 0) g_trace_on = 1;
    }

    /* Find the input file (first arg that isn't a flag) */
    const char* input_file = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--emit-eskb") == 0) { i++; continue; }
        if (strcmp(argv[i], "--trace") == 0) continue;
        input_file = argv[i];
        break;
    }

    if (input_file) {
        /* Check if input is a .eskb file — load and execute directly */
        size_t flen = strlen(input_file);
        if (flen > 5 && strcmp(input_file + flen - 5, ".eskb") == 0) {
            EskbModule mod;
            if (eskb_load_file(input_file, &mod) == 0) {
                /* Convert to VM FuncChunk format and execute */
                printf("  [ESKB] Loaded %d instructions, %d constants from %s\n",
                       mod.code_len, mod.n_constants, input_file);
                if (mod.has_debug) printf("  [ESKB] Source: %s\n", mod.source_file);

                /* Build a FuncChunk from the loaded module */
                FuncChunk chunk = {0};
                for (int i = 0; i < mod.code_len && i < MAX_CODE; i++) {
                    chunk.code[i] = (Instr){mod.opcodes[i], mod.operands[i]};
                }
                chunk.code_len = mod.code_len;
                for (int i = 0; i < mod.n_constants && i < MAX_CONSTS; i++) {
                    switch (mod.const_types[i]) {
                    case ESKB_CONST_NIL:
                        chunk.constants[i] = (Value){.type=VAL_NIL, .as.i=0};
                        break;
                    case ESKB_CONST_INT64:
                        chunk.constants[i] = (Value){.type=VAL_INT, .as.i=mod.const_ints[i]};
                        break;
                    case ESKB_CONST_F64:
                        chunk.constants[i] = (Value){.type=VAL_FLOAT, .as.f=mod.const_floats[i]};
                        break;
                    case ESKB_CONST_BOOL:
                        chunk.constants[i] = (Value){.type=VAL_BOOL, .as.b=(int)mod.const_ints[i]};
                        break;
                    default:
                        chunk.constants[i] = (Value){.type=VAL_INT, .as.i=mod.const_ints[i]};
                        break;
                    }
                }
                chunk.n_constants = mod.n_constants;
                execute_chunk(&chunk);
                eskb_module_free(&mod);
            } else {
                printf("ERROR: failed to load ESKB file %s\n", input_file);
                return 1;
            }
        } else {
            /* Compile .esk source file */
            FILE* f = fopen(input_file, "r");
            if (!f) { printf("ERROR: cannot open %s\n", input_file); return 1; }
            fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
            char* src = (char*)malloc(len + 1);
            fread(src, 1, len, f); src[len] = 0; fclose(f);
            printf("  Source: %s\n", input_file);
            g_source_file_path = input_file;
            compile_and_run(src);
            free(src);
        }
    } else {
        /* Built-in tests */
        printf("  Test 1: (display (+ 3 5))\n");
        compile_and_run("(display (+ 3 5))");

        printf("\n  Test 2: (let ((x 10)) (display (if (> x 5) (* x 2) x)))\n");
        compile_and_run("(let ((x 10)) (display (if (> x 5) (* x 2) x)))");

        printf("\n  Test 3: factorial\n");
        compile_and_run(
            "(define (factorial n)\n"
            "  (if (= n 0) 1 (* n (factorial (- n 1)))))\n"
            "(display (factorial 10))\n");

        printf("\n  Test 4: fibonacci\n");
        compile_and_run(
            "(define (fib n)\n"
            "  (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))\n"
            "(display (fib 10))\n");

        printf("\n  Test 5: map with lambda\n");
        compile_and_run(
            "(define (map f lst)\n"
            "  (if (null? lst) '()\n"
            "      (cons (f (car lst)) (map f (cdr lst)))))\n"
            "(display (map (lambda (x) (* x x)) (list 1 2 3 4 5)))\n");

        /* ── Edge Case Tests ── */
        printf("\n  Edge case tests:\n");

        /* Modulo with negative numbers (R7RS floored modulo) */
        printf("\n  Test 6: modulo-neg\n");
        compile_and_run("(display (modulo -5 3))");  /* should output 1 */

        /* Nested let expressions */
        printf("\n  Test 7: nested-let\n");
        compile_and_run("(display (let ((x 10)) (let ((y (+ x 5))) (* x y))))");  /* should output 150 */

        /* Higher-order: compose */
        printf("\n  Test 8: compose\n");
        compile_and_run(
            "(define (compose f g) (lambda (x) (f (g x))))\n"
            "(define (add1 x) (+ x 1))\n"
            "(define (double x) (* x 2))\n"
            "(display ((compose double add1) 5))");  /* should output 12 */

        /* Tail recursion: should not stack overflow */
        printf("\n  Test 9: deep-tail-call\n");
        compile_and_run(
            "(define (count n) (if (= n 0) 0 (count (- n 1))))\n"
            "(display (count 100000))");  /* should output 0 */

        /* Boolean edge cases */
        printf("\n  Test 10: bool-edge\n");
        compile_and_run("(display (if '() 1 2))");  /* R7RS: '() is truthy; should output 1 */

        /* Constant folding verification */
        printf("\n  Test 11: const-fold\n");
        compile_and_run("(display (+ (* 3 4) (- 10 5)))");  /* should output 17, folded at compile time */

        printf("\n  All tests complete.\n");
    }

    printf("\n=== Compilation complete ===\n");
    return 0;
}
