static void state_init(State* st) {
    memset(st, 0, sizeof(State));
    st->s[S_OUTPUT] = -1.0f;
    st->s[S_CUR_CLOSURE] = -100.0f;
}

/**
 * @brief Ground-truth reference interpreter: executes one VM instruction
 *        step as a plain C switch over the opcode, updating @p next from
 *        @p cur (stack registers, memory, AD tape state, arena, etc.).
 *
 * This is the "correct by construction" implementation that the simulated
 * transformer (execute via layerN_ffn()) and the matrix forward pass
 * (forward_with_weights()) are both verified against.
 */
static void execute_step(const State* cur, const Instr* prog, int n_instr, State* next) {
    memcpy(next, cur, sizeof(State));
    next->s[S_OUTPUT] = -1.0f;
    next->s[S_HAS_OUT] = 0;
    /* Clear intermediates: Zone A (16-31), Zone B cursor (39-47),
     * Zone D (112-127), and arena op transients. */
    for (int i = S_OPCODE; i <= S_ABS_DELTA; i++) next->s[i] = 0;
    for (int i = S_AD_CUR_OP; i <= S_AD_RIGHT_VALUE; i++) next->s[i] = 0;
    for (int i = S_AD_IS_FORWARD; i <= S_AD_SPARE8; i++) next->s[i] = 0;
    for (int i = S_ARENA_TRANSIENT_START; i <= S_ARENA_TRANSIENT_END; i++) next->s[i] = 0;

    int pc = (int)cur->s[S_PC];
    if (pc < 0 || pc >= n_instr || cur->s[S_HALT] > 0.5f) { next->s[S_HALT] = 1; return; }

    float tos = cur->s[S_TOS], sos = cur->s[S_SOS];
    float r2 = cur->s[S_R2], r3 = cur->s[S_R3];
    float operand = (float)prog[pc].operand;
    int addr;

    /* Save current type tags for shifting */
    float tt_tos = cur->s[S_TYPE_TOS], tt_sos = cur->s[S_TYPE_SOS];
    float tt_r2  = cur->s[S_TYPE_R2],  tt_r3  = cur->s[S_TYPE_R3];

    switch (prog[pc].op) {
    case OP_NOP:    next->s[S_PC]=pc+1; break;
    case OP_CONST:  next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=operand; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; next->s[S_TYPE_TOS]=TYPE_NUMBER;
        break;
    case OP_NIL:    next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=-1; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; next->s[S_TYPE_TOS]=TYPE_NIL;
        break;
    case OP_TRUE:   next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=1; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_FALSE:  next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_ADD:    next->s[S_TOS]=tos+sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_SUB:    next->s[S_TOS]=sos-tos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_MUL:    next->s[S_TOS]=tos*sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_NEG:    next->s[S_TOS]=-tos; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER;
        break;
    case OP_ABS:    next->s[S_TOS]=fabsf(tos); next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER;
        break;
    case OP_EQ:     next->s[S_TOS]=(tos==sos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_LT:     next->s[S_TOS]=(sos<tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_GT:     next->s[S_TOS]=(sos>tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_LE:     next->s[S_TOS]=(sos<=tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_GE:     next->s[S_TOS]=(sos>=tos)?1.0f:0.0f; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_DIV:
        if (tos == 0) { next->s[S_HALT] = 1; }
        else { next->s[S_TOS]=sos/tos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER; }
        break;
    case OP_MOD:
        if (tos == 0) { next->s[S_HALT] = 1; }
        else { float r=fmodf(sos,tos); if(r!=0&&((r>0)!=(tos>0)))r+=tos; next->s[S_TOS]=r; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER; }
        break;
    case OP_NOT:    next->s[S_TOS]=(tos==0)?1.0f:0.0f; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_POP:    next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_DUP:    next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; /* TOS type stays */
        break;
    case OP_SWAP:   next->s[S_TOS]=sos; next->s[S_SOS]=tos; next->s[S_PC]=pc+1; /* depth, R2, R3 unchanged */
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_tos;
        break;
    case OP_GET_LOCAL:
        addr=(int)operand;
        if(addr>=0&&addr<MEM_SIZE) {
            next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos;
            next->s[S_TOS]=cur->s[S_MEM0+addr]; next->s[S_DEPTH]=cur->s[S_DEPTH]+1;
            next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos;
            next->s[S_TYPE_TOS]=TYPE_NUMBER; /* locals are untyped, assume number */
        }
        next->s[S_PC]=pc+1; break;
    case OP_SET_LOCAL:
        addr=(int)operand;
        if(addr>=0&&addr<MEM_SIZE) next->s[S_MEM0+addr]=tos;
        next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        next->s[S_PC]=pc+1; break;
    case OP_GET_UPVALUE:
        addr=(int)operand;
        next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos;
        next->s[S_TOS]=(addr>=0&&addr<MEM_SIZE)?cur->s[S_MEM0+addr]:0;
        next->s[S_TYPE_TOS]=TYPE_NUMBER;
        if (addr >= 0 && addr < MEM_SIZE && cur->s[S_CUR_CLOSURE] >= 0.0f) {
            int upcell = (int)(cur->s[S_CUR_CLOSURE] + 1.0f + (float)addr);
            if (upcell >= 0 && upcell < ARENA_CELLS) {
                next->s[S_TOS]=ARENA_FIELD(cur->s, upcell, ARENA_F_CAR_VAL);
                next->s[S_TYPE_TOS]=ARENA_FIELD(cur->s, upcell, ARENA_F_CAR_TYPE);
            }
        }
        next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos;
        break;
    case OP_SET_UPVALUE:
        addr=(int)operand;
        if(addr>=0&&addr<MEM_SIZE) next->s[S_MEM0+addr]=tos;
        if (addr >= 0 && addr < MEM_SIZE && cur->s[S_CUR_CLOSURE] >= 0.0f) {
            int upcell = (int)(cur->s[S_CUR_CLOSURE] + 1.0f + (float)addr);
            if (upcell >= 0 && upcell < ARENA_CELLS) {
                ARENA_FIELD(next->s, upcell, ARENA_F_CAR_VAL) = tos;
                ARENA_FIELD(next->s, upcell, ARENA_F_CAR_TYPE) = tt_tos;
            }
        }
        next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        next->s[S_PC]=pc+1; break;
    case OP_CLOSE_UPVALUE:
        addr=(int)operand;
        if (addr >= 0 && addr < MEM_SIZE && cur->s[S_CUR_CLOSURE] >= 0.0f) {
            int upcell = (int)(cur->s[S_CUR_CLOSURE] + 1.0f + (float)addr);
            if (upcell >= 0 && upcell < ARENA_CELLS) {
                ARENA_FIELD(next->s, upcell, ARENA_F_CAR_VAL) = cur->s[S_MEM0+addr];
                ARENA_FIELD(next->s, upcell, ARENA_F_CAR_TYPE) = TYPE_NUMBER;
            }
        }
        next->s[S_PC]=pc+1; break;
    case OP_OPEN_CLOSURE:
        next->s[S_CUR_CLOSURE]=tos;
        next->s[S_PC]=pc+1; break;
    case OP_CALL:   /* Set IS_CALL for exec loop to handle frame management */
        next->s[S_IS_CALL]=1; next->s[S_PC]=pc+1; break;
    case OP_TAIL_CALL: {
        int argc = operand;
        if (argc < 0) argc = 0;
        if (argc > MEM_SIZE) argc = MEM_SIZE;
        float args[4] = {sos, r2, r3, 0};
        for (int i = 0; i < MEM_SIZE; i++) next->s[S_MEM0+i] = 0;
        for (int i = 0; i < argc && i < MEM_SIZE; i++)
            next->s[S_MEM0+i] = args[i];
        next->s[S_PC]=tos;
        next->s[S_TOS]=0; next->s[S_SOS]=0; next->s[S_R2]=0; next->s[S_R3]=0;
        next->s[S_DEPTH]=cur->s[S_DEPTH]-(1+argc);
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=TYPE_NUMBER;
        next->s[S_TYPE_R2]=TYPE_NUMBER; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    }
    case OP_RETURN: /* Set IS_RET for exec loop to handle frame restore */
        next->s[S_IS_RET]=1; next->s[S_PC]=pc+1; break;
    case OP_JUMP:   next->s[S_PC]=operand; break;
    case OP_JUMP_IF_FALSE:
        next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        next->s[S_PC]=(tos==0)?operand:(float)(pc+1); break;
    case OP_LOOP:   next->s[S_PC]=operand; break;
    case OP_PRINT:  next->s[S_OUTPUT]=tos; next->s[S_HAS_OUT]=1; next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0; next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    case OP_HALT:   next->s[S_HALT]=1; break;

    /* Stage-2 arena-backed pair ops. The transformer VM stores cons cells
     * in a bounded bump arena inside the state vector; stack values are
     * arena cell indices, not host pointers. */
    case OP_CONS: {
        int cell = (int)cur->s[S_ARENA_NEXT];
        if (cell < 0 || cell >= ARENA_CELLS) { next->s[S_HALT]=1; break; }
        ARENA_FIELD(next->s, cell, ARENA_F_KIND) = ARENA_KIND_PAIR;
        ARENA_FIELD(next->s, cell, ARENA_F_CAR_VAL) = sos;
        ARENA_FIELD(next->s, cell, ARENA_F_CDR_VAL) = tos;
        ARENA_FIELD(next->s, cell, ARENA_F_CAR_TYPE) = tt_sos;
        ARENA_FIELD(next->s, cell, ARENA_F_CDR_TYPE) = tt_tos;
        next->s[S_ARENA_NEXT] = (float)(cell + 1);
        next->s[S_TOS]=(float)cell; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0;
        next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_PAIR; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    }
    case OP_CAR: {
        int cell = (int)tos;
        next->s[S_TOS]=0; next->s[S_TYPE_TOS]=TYPE_NUMBER;
        if (cell >= 0 && cell < ARENA_CELLS) {
            next->s[S_TOS]=ARENA_FIELD(cur->s, cell, ARENA_F_CAR_VAL);
            next->s[S_TYPE_TOS]=ARENA_FIELD(cur->s, cell, ARENA_F_CAR_TYPE);
        }
        next->s[S_PC]=pc+1;
        break;
    }
    case OP_CDR: {
        int cell = (int)tos;
        next->s[S_TOS]=0; next->s[S_TYPE_TOS]=TYPE_NUMBER;
        if (cell >= 0 && cell < ARENA_CELLS) {
            next->s[S_TOS]=ARENA_FIELD(cur->s, cell, ARENA_F_CDR_VAL);
            next->s[S_TYPE_TOS]=ARENA_FIELD(cur->s, cell, ARENA_F_CDR_TYPE);
        }
        next->s[S_PC]=pc+1;
        break;
    }
    case OP_SET_CAR: {
        int cell = (int)sos;
        if (cell >= 0 && cell < ARENA_CELLS) {
            ARENA_FIELD(next->s, cell, ARENA_F_CAR_VAL) = tos;
            ARENA_FIELD(next->s, cell, ARENA_F_CAR_TYPE) = tt_tos;
        }
        next->s[S_TOS]=r2; next->s[S_SOS]=r3; next->s[S_R2]=0; next->s[S_R3]=0;
        next->s[S_DEPTH]=cur->s[S_DEPTH]-2; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=tt_r2; next->s[S_TYPE_SOS]=tt_r3; next->s[S_TYPE_R2]=TYPE_NUMBER; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    }
    case OP_SET_CDR: {
        int cell = (int)sos;
        if (cell >= 0 && cell < ARENA_CELLS) {
            ARENA_FIELD(next->s, cell, ARENA_F_CDR_VAL) = tos;
            ARENA_FIELD(next->s, cell, ARENA_F_CDR_TYPE) = tt_tos;
        }
        next->s[S_TOS]=r2; next->s[S_SOS]=r3; next->s[S_R2]=0; next->s[S_R3]=0;
        next->s[S_DEPTH]=cur->s[S_DEPTH]-2; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=tt_r2; next->s[S_TYPE_SOS]=tt_r3; next->s[S_TYPE_R2]=TYPE_NUMBER; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    }
    case OP_CLOSURE: {
        int cell = (int)cur->s[S_ARENA_NEXT];
        if (cell < 0 || cell + 1 + MEM_SIZE > ARENA_CELLS) { next->s[S_HALT]=1; break; }
        ARENA_FIELD(next->s, cell, ARENA_F_KIND) = ARENA_KIND_CLOSURE;
        ARENA_FIELD(next->s, cell, ARENA_F_CAR_VAL) = (float)operand;
        ARENA_FIELD(next->s, cell, ARENA_F_CDR_VAL) = (float)MEM_SIZE;
        ARENA_FIELD(next->s, cell, ARENA_F_CAR_TYPE) = TYPE_NUMBER;
        ARENA_FIELD(next->s, cell, ARENA_F_CDR_TYPE) = TYPE_NUMBER;
        next->s[S_ARENA_NEXT] = (float)(cell + 1 + MEM_SIZE);
        next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=(float)cell;
        next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_CLOSURE; next->s[S_TYPE_SOS]=tt_tos;
        next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_R3]=tt_r2;
        break;
    }
    case OP_VEC_CREATE: {
        int count = operand;
        if (count < 0 || count > ARENA_MAX_INLINE_VECTOR ||
            (int)cur->s[S_ARENA_NEXT] + count >= ARENA_CELLS) {
            next->s[S_HALT]=1;
            break;
        }
        int base = (int)cur->s[S_ARENA_NEXT];
        float vals[4] = {tos, sos, r2, r3};
        float types[4] = {tt_tos, tt_sos, tt_r2, tt_r3};
        ARENA_FIELD(next->s, base, ARENA_F_KIND) = ARENA_KIND_VECTOR;
        ARENA_FIELD(next->s, base, ARENA_F_CAR_VAL) = (float)count;
        ARENA_FIELD(next->s, base, ARENA_F_CDR_VAL) = (float)(base + 1);
        ARENA_FIELD(next->s, base, ARENA_F_CAR_TYPE) = TYPE_NUMBER;
        ARENA_FIELD(next->s, base, ARENA_F_CDR_TYPE) = TYPE_NUMBER;
        for (int i = 0; i < count; i++) {
            int elem_cell = base + 1 + i;
            int src = count - 1 - i;
            ARENA_FIELD(next->s, elem_cell, ARENA_F_KIND) = ARENA_KIND_VEC_ELEM;
            ARENA_FIELD(next->s, elem_cell, ARENA_F_CAR_VAL) = vals[src];
            ARENA_FIELD(next->s, elem_cell, ARENA_F_CDR_VAL) = (float)(elem_cell + 1);
            ARENA_FIELD(next->s, elem_cell, ARENA_F_CAR_TYPE) = types[src];
            ARENA_FIELD(next->s, elem_cell, ARENA_F_CDR_TYPE) = TYPE_NUMBER;
        }
        next->s[S_ARENA_NEXT] = (float)(base + 1 + count);
        next->s[S_TOS]=(float)base; next->s[S_SOS]=0; next->s[S_R2]=0; next->s[S_R3]=0;
        next->s[S_DEPTH]=cur->s[S_DEPTH] - (float)(count - 1); next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_VECTOR; next->s[S_TYPE_SOS]=TYPE_NUMBER;
        next->s[S_TYPE_R2]=TYPE_NUMBER; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    }
    case OP_VEC_REF:
    case OP_STR_REF: {
        int base = (int)sos;
        int idx = (int)tos;
        next->s[S_TOS]=0; next->s[S_TYPE_TOS]=TYPE_NUMBER;
        if (base >= 0 && idx >= 0 && idx < ARENA_MAX_INLINE_VECTOR) {
            int elem_cell = base + 1 + idx;
            if (elem_cell >= 0 && elem_cell < ARENA_CELLS) {
                next->s[S_TOS]=ARENA_FIELD(cur->s, elem_cell, ARENA_F_CAR_VAL);
                next->s[S_TYPE_TOS]=ARENA_FIELD(cur->s, elem_cell, ARENA_F_CAR_TYPE);
            }
        }
        next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0;
        next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    }
    case OP_VEC_SET: {
        int base = (int)r2;
        int idx = (int)sos;
        if (base >= 0 && idx >= 0 && idx < ARENA_MAX_INLINE_VECTOR) {
            int elem_cell = base + 1 + idx;
            if (elem_cell >= 0 && elem_cell < ARENA_CELLS) {
                ARENA_FIELD(next->s, elem_cell, ARENA_F_CAR_VAL) = tos;
                ARENA_FIELD(next->s, elem_cell, ARENA_F_CAR_TYPE) = tt_tos;
            }
        }
        next->s[S_TOS]=r3; next->s[S_SOS]=0; next->s[S_R2]=0; next->s[S_R3]=0;
        next->s[S_DEPTH]=cur->s[S_DEPTH]-3; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=tt_r3; next->s[S_TYPE_SOS]=TYPE_NUMBER; next->s[S_TYPE_R2]=TYPE_NUMBER; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    }
    case OP_VEC_LEN:
    case OP_STR_LEN: {
        int base = (int)tos;
        next->s[S_TOS]=0; next->s[S_TYPE_TOS]=TYPE_NUMBER;
        if (base >= 0 && base < ARENA_CELLS)
            next->s[S_TOS]=ARENA_FIELD(cur->s, base, ARENA_F_CAR_VAL);
        next->s[S_PC]=pc+1;
        break;
    }
    case OP_PACK_REST: {
        int n_fixed = (int)operand;
        if (n_fixed < 0) n_fixed = 0;
        if (n_fixed > MEM_SIZE) n_fixed = MEM_SIZE;
        int count = MEM_SIZE - n_fixed;
        if (count > 0) {
            int base = (int)cur->s[S_ARENA_NEXT];
            if (base < 0 || base + count > ARENA_CELLS) { next->s[S_HALT]=1; break; }
            for (int j = 0; j < count; j++) {
                int cell = base + j;
                ARENA_FIELD(next->s, cell, ARENA_F_KIND) = ARENA_KIND_PAIR;
                ARENA_FIELD(next->s, cell, ARENA_F_CAR_VAL) = cur->s[S_MEM0+n_fixed+j];
                ARENA_FIELD(next->s, cell, ARENA_F_CAR_TYPE) = TYPE_NUMBER;
                if (j + 1 < count) {
                    ARENA_FIELD(next->s, cell, ARENA_F_CDR_VAL) = (float)(cell + 1);
                    ARENA_FIELD(next->s, cell, ARENA_F_CDR_TYPE) = TYPE_PAIR;
                } else {
                    ARENA_FIELD(next->s, cell, ARENA_F_CDR_VAL) = -1.0f;
                    ARENA_FIELD(next->s, cell, ARENA_F_CDR_TYPE) = TYPE_NIL;
                }
            }
            next->s[S_ARENA_NEXT] = (float)(base + count);
            next->s[S_MEM0+n_fixed] = (float)base;
        }
        next->s[S_PC]=pc+1;
        break;
    }
    case OP_PUSH_HANDLER:
        next->s[S_EXC_DEPTH]=cur->s[S_EXC_DEPTH]+1;
        next->s[S_PC]=pc+1;
        break;
    case OP_POP_HANDLER:
        next->s[S_EXC_DEPTH]=cur->s[S_EXC_DEPTH]-1;
        next->s[S_PC]=pc+1;
        break;
    case OP_GET_EXN:
        next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos; next->s[S_TOS]=g_current_exn;
        next->s[S_TYPE_R3]=tt_r2; next->s[S_TYPE_R2]=tt_sos; next->s[S_TYPE_SOS]=tt_tos; next->s[S_TYPE_TOS]=TYPE_NUMBER;
        next->s[S_DEPTH]=cur->s[S_DEPTH]+1; next->s[S_PC]=pc+1;
        break;
    case OP_WIND_PUSH:
        next->s[S_WIND_DEPTH]=cur->s[S_WIND_DEPTH]+1;
        next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0;
        next->s[S_TYPE_TOS]=tt_sos; next->s[S_TYPE_SOS]=tt_r2; next->s[S_TYPE_R2]=tt_r3; next->s[S_TYPE_R3]=TYPE_NUMBER;
        next->s[S_DEPTH]=cur->s[S_DEPTH]-1; next->s[S_PC]=pc+1;
        break;
    case OP_WIND_POP:
        next->s[S_WIND_DEPTH]=cur->s[S_WIND_DEPTH]-1;
        next->s[S_PC]=pc+1;
        break;

    /* Bounded continuation slice. The transformer VM records the directly
     * modeled continuation state into four contiguous arena cells. This covers
     * artifact-shape escape continuations without using host heap pointers. */
    case OP_CALLCC: {
        int base = (int)cur->s[S_ARENA_NEXT];
        if (base < 0 || base + ARENA_CONT_CELLS > ARENA_CELLS) {
            next->s[S_HALT]=1;
            break;
        }

        float cont_payload[ARENA_CONT_CELLS][ARENA_CELL_FIELDS] = {
            {ARENA_KIND_PAIR, (float)(pc + 1), cur->s[S_DEPTH] - 1.0f, sos, r2},
            {ARENA_KIND_PAIR, r3, 0.0f, tt_sos, tt_r2},
            {ARENA_KIND_PAIR, tt_r3, TYPE_NUMBER, cur->s[S_MEM0], cur->s[S_MEM1]},
            {ARENA_KIND_PAIR, cur->s[S_MEM2], cur->s[S_MEM3], cur->s[S_WIND_DEPTH], 0.0f},
        };
        for (int i = 0; i < ARENA_CONT_CELLS; i++)
            for (int f = 0; f < ARENA_CELL_FIELDS; f++)
                ARENA_FIELD(next->s, base + i, f) = cont_payload[i][f];

        next->s[S_ARENA_NEXT]=(float)(base + ARENA_CONT_CELLS);
        next->s[S_MEM0]=(float)base; next->s[S_MEM1]=0; next->s[S_MEM2]=0; next->s[S_MEM3]=0;
        next->s[S_PC]=tos;
        next->s[S_TOS]=0; next->s[S_SOS]=0; next->s[S_R2]=0; next->s[S_R3]=0;
        next->s[S_DEPTH]=0;
        next->s[S_TYPE_TOS]=TYPE_NUMBER; next->s[S_TYPE_SOS]=TYPE_NUMBER;
        next->s[S_TYPE_R2]=TYPE_NUMBER; next->s[S_TYPE_R3]=TYPE_NUMBER;
        break;
    }
    case OP_INVOKE_CC: {
        int base = (int)sos;
        if (base >= 0 && base + ARENA_CONT_CELLS <= ARENA_CELLS) {
            float retval = tos;
            float rettype = tt_tos;
            next->s[S_PC]   = ARENA_FIELD(cur->s, base + 0, ARENA_F_CAR_VAL);
            next->s[S_DEPTH]= ARENA_FIELD(cur->s, base + 0, ARENA_F_CDR_VAL) + 1.0f;
            next->s[S_MEM0] = ARENA_FIELD(cur->s, base + 2, ARENA_F_CAR_TYPE);
            next->s[S_MEM1] = ARENA_FIELD(cur->s, base + 2, ARENA_F_CDR_TYPE);
            next->s[S_MEM2] = ARENA_FIELD(cur->s, base + 3, ARENA_F_CAR_VAL);
            next->s[S_MEM3] = ARENA_FIELD(cur->s, base + 3, ARENA_F_CDR_VAL);
            next->s[S_TOS]  = retval;
            next->s[S_SOS]  = ARENA_FIELD(cur->s, base + 0, ARENA_F_CAR_TYPE);
            next->s[S_R2]   = ARENA_FIELD(cur->s, base + 0, ARENA_F_CDR_TYPE);
            next->s[S_R3]   = ARENA_FIELD(cur->s, base + 1, ARENA_F_CAR_VAL);
            next->s[S_TYPE_TOS] = rettype;
            next->s[S_TYPE_SOS] = ARENA_FIELD(cur->s, base + 1, ARENA_F_CAR_TYPE);
            next->s[S_TYPE_R2]  = ARENA_FIELD(cur->s, base + 1, ARENA_F_CDR_TYPE);
            next->s[S_TYPE_R3]  = ARENA_FIELD(cur->s, base + 2, ARENA_F_CAR_VAL);
            next->s[S_WIND_DEPTH] = ARENA_FIELD(cur->s, base + 3, ARENA_F_CAR_TYPE);
        } else {
            next->s[S_HALT]=1;
        }
        break;
    }

    /* Stage-1 VM-as-transformer memory ops: directly encodable predicates
     * and stack cleanup. These used to set IS_NATIVE and round-trip through
     * exec_loop_postprocess; keeping them here makes the reference path match
     * the simulated/matrix weight implementation. */
    case OP_NULL_P:
        next->s[S_TOS]=(tt_tos == TYPE_NIL) ? 1.0f : 0.0f; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_PAIR_P:
        next->s[S_TOS]=(tt_tos == TYPE_PAIR) ? 1.0f : 0.0f; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_NUM_P:
        next->s[S_TOS]=(tt_tos == TYPE_NUMBER) ? 1.0f : 0.0f; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_STR_P:
        next->s[S_TOS]=(tt_tos == TYPE_STRING) ? 1.0f : 0.0f; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_BOOL_P:
        next->s[S_TOS]=(tt_tos == TYPE_BOOL) ? 1.0f : 0.0f; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_PROC_P:
        next->s[S_TOS]=(tt_tos == TYPE_CLOSURE) ? 1.0f : 0.0f; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_VEC_P:
        next->s[S_TOS]=(tt_tos == TYPE_VECTOR) ? 1.0f : 0.0f; next->s[S_PC]=pc+1;
        next->s[S_TYPE_TOS]=TYPE_BOOL;
        break;
    case OP_POPN: {
        int count = (int)operand;
        if (count < 0) count = 0;
        float regs[4] = {tos, sos, r2, r3};
        float types[4] = {tt_tos, tt_sos, tt_r2, tt_r3};
        for (int i = 1; i < 4; i++) {
            int src = i + count;
            if (src < 4) { regs[i] = regs[src]; types[i] = types[src]; }
            else { regs[i] = 0; types[i] = TYPE_NUMBER; }
        }
        next->s[S_TOS]=regs[0]; next->s[S_SOS]=regs[1]; next->s[S_R2]=regs[2]; next->s[S_R3]=regs[3];
        next->s[S_TYPE_TOS]=types[0]; next->s[S_TYPE_SOS]=types[1]; next->s[S_TYPE_R2]=types[2]; next->s[S_TYPE_R3]=types[3];
        next->s[S_DEPTH]=cur->s[S_DEPTH]-count; next->s[S_PC]=pc+1;
        break;
    }
    case OP_VOID:
        next->s[S_PC]=pc+1;
        break;

    /* ── AD Forward Ops: record nodes on the embedded tape ── */
    case OP_AD_VAR: { /* (ad-var value) → push tape index */
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        if (tlen < AD_MAX_TAPE) {
            AD_NODE(next->s, tlen, AD_F_OP) = AD_OP_VAR;
            AD_NODE(next->s, tlen, AD_F_VALUE) = operand;
            AD_NODE(next->s, tlen, AD_F_GRAD) = 0;
            AD_NODE(next->s, tlen, AD_F_LEFT) = -1;
            AD_NODE(next->s, tlen, AD_F_RIGHT) = -1;
            AD_NODE(next->s, tlen, AD_F_SAVED) = 0;
            /* Push tape index onto register stack */
            next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos;
            next->s[S_TOS]=(float)tlen;
            next->s[S_DEPTH]=cur->s[S_DEPTH]+1;
            next->s[S_AD_TAPE_LEN]=(float)(tlen+1);
            next->s[S_AD_MODE]=1;
        }
        next->s[S_PC]=pc+1; break;
    }
    case OP_AD_CONST: { /* (ad-const value) → push tape index */
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        if (tlen < AD_MAX_TAPE) {
            AD_NODE(next->s, tlen, AD_F_OP) = AD_OP_CONST;
            AD_NODE(next->s, tlen, AD_F_VALUE) = operand;
            AD_NODE(next->s, tlen, AD_F_GRAD) = 0;
            AD_NODE(next->s, tlen, AD_F_LEFT) = -1;
            AD_NODE(next->s, tlen, AD_F_RIGHT) = -1;
            AD_NODE(next->s, tlen, AD_F_SAVED) = 0;
            next->s[S_R3]=r2; next->s[S_R2]=sos; next->s[S_SOS]=tos;
            next->s[S_TOS]=(float)tlen;
            next->s[S_DEPTH]=cur->s[S_DEPTH]+1;
            next->s[S_AD_TAPE_LEN]=(float)(tlen+1);
            next->s[S_AD_MODE]=1;
        }
        next->s[S_PC]=pc+1; break;
    }
    case OP_AD_ADD: case OP_AD_SUB: case OP_AD_MUL:
    case OP_AD_DIV: case OP_AD_POW: { /* binary: TOS=right_idx, SOS=left_idx */
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        int li = (int)sos, ri = (int)tos;
        if (tlen < AD_MAX_TAPE && li >= 0 && li < tlen && ri >= 0 && ri < tlen) {
            float lv = AD_NODE(cur->s, li, AD_F_VALUE);
            float rv = AD_NODE(cur->s, ri, AD_F_VALUE);
            float val = 0, saved = 0;
            float op_type = 0;
            OpCode cur_op = prog[pc].op;
            if (cur_op == OP_AD_ADD) { val = lv + rv; op_type = AD_OP_ADD; }
            else if (cur_op == OP_AD_SUB) { val = lv - rv; op_type = AD_OP_SUB; }
            else if (cur_op == OP_AD_MUL) { val = lv * rv; op_type = AD_OP_MUL; }
            else if (cur_op == OP_AD_DIV) {
                float safe_rv = fabsf(rv) > 1e-15f ? rv : (rv < 0 ? -1e-15f : 1e-15f);
                val = lv / safe_rv;
                saved = 1.0f / safe_rv;
                op_type = AD_OP_DIV;
            } else {
                float safe_lv = lv > 1e-15f ? lv : 1e-15f;
                val = powf(safe_lv, rv);
                saved = rv * powf(safe_lv, rv - 1.0f);
                op_type = AD_OP_POW;
            }
            AD_NODE(next->s, tlen, AD_F_OP) = op_type;
            AD_NODE(next->s, tlen, AD_F_VALUE) = val;
            AD_NODE(next->s, tlen, AD_F_GRAD) = 0;
            AD_NODE(next->s, tlen, AD_F_LEFT) = (float)li;
            AD_NODE(next->s, tlen, AD_F_RIGHT) = (float)ri;
            AD_NODE(next->s, tlen, AD_F_SAVED) = saved;
            /* Pop two, push tape index */
            next->s[S_TOS]=(float)tlen; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0;
            next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
            next->s[S_AD_TAPE_LEN]=(float)(tlen+1);
        }
        next->s[S_PC]=pc+1; break;
    }
    case OP_AD_NEG: case OP_AD_ABS: case OP_AD_RELU:
    case OP_AD_SIGMOID: case OP_AD_TANH:
    case OP_AD_EXP: case OP_AD_LOG: case OP_AD_SQRT:
    case OP_AD_SIN: case OP_AD_COS: { /* unary: TOS=input_idx */
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        int ii = (int)tos;
        if (tlen < AD_MAX_TAPE && ii >= 0 && ii < tlen) {
            float iv = AD_NODE(cur->s, ii, AD_F_VALUE);
            float val = 0, op_type = 0, saved = 0;
            switch (prog[pc].op) {
                case OP_AD_NEG:     val = -iv;                          op_type = AD_OP_NEG;     saved = -1.0f; break;
                case OP_AD_ABS:     val = fabsf(iv);                    op_type = AD_OP_ABS;     saved = (iv > 0) ? 1.0f : (iv < 0) ? -1.0f : 0.0f; break;
                case OP_AD_RELU:    val = iv > 0 ? iv : 0;             op_type = AD_OP_RELU;    saved = (iv > 0) ? 1.0f : 0.0f; break;
                case OP_AD_SIGMOID: val = 1.0f/(1.0f+expf(-iv));       op_type = AD_OP_SIGMOID; saved = val * (1.0f - val); break;
                case OP_AD_TANH:    val = tanhf(iv);                    op_type = AD_OP_TANH;    saved = 1.0f - val * val; break;
                case OP_AD_EXP:     val = expf(iv);                     op_type = AD_OP_EXP;     saved = val; break;
                case OP_AD_LOG:     val = logf(iv);                     op_type = AD_OP_LOG;     saved = 1.0f / (fabsf(iv) > 1e-15f ? iv : 1e-15f); break;
                case OP_AD_SQRT:    val = sqrtf(iv);                    op_type = AD_OP_SQRT;    saved = 1.0f / (2.0f * (fabsf(val) > 1e-15f ? val : 1e-15f)); break;
                case OP_AD_SIN:     val = sinf(iv);                     op_type = AD_OP_SIN;     saved = cosf(iv); break;
                case OP_AD_COS:     val = cosf(iv);                     op_type = AD_OP_COS;     saved = -sinf(iv); break;
                default: break;
            }
            AD_NODE(next->s, tlen, AD_F_OP) = op_type;
            AD_NODE(next->s, tlen, AD_F_VALUE) = val;
            AD_NODE(next->s, tlen, AD_F_GRAD) = 0;
            AD_NODE(next->s, tlen, AD_F_LEFT) = (float)ii;
            AD_NODE(next->s, tlen, AD_F_RIGHT) = -1;
            AD_NODE(next->s, tlen, AD_F_SAVED) = saved;
            /* Replace TOS with tape index */
            next->s[S_TOS]=(float)tlen;
            next->s[S_AD_TAPE_LEN]=(float)(tlen+1);
        }
        next->s[S_PC]=pc+1; break;
    }
    case OP_AD_BACKWARD: { /* Start backward pass from TOS (output node index) */
        int output_idx = (int)tos;
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        if (output_idx >= 0 && output_idx < tlen) {
            next->s[S_AD_MODE] = 2;
            next->s[S_AD_CURSOR] = (float)output_idx;
            next->s[S_AD_IS_BACKWARD] = 1;
            /* Seed output gradient = 1.0 */
            AD_NODE(next->s, output_idx, AD_F_GRAD) = 1.0f;
            /* Pop the output index from stack */
            next->s[S_TOS]=sos; next->s[S_SOS]=r2; next->s[S_R2]=r3; next->s[S_R3]=0;
            next->s[S_DEPTH]=cur->s[S_DEPTH]-1;
        }
        next->s[S_PC]=pc+1; break;
    }
    case OP_AD_GRAD: { /* Push gradient of TOS (node index) onto stack */
        int ni = (int)tos;
        int tlen = (int)cur->s[S_AD_TAPE_LEN];
        float grad = 0;
        if (ni >= 0 && ni < tlen) grad = AD_NODE(cur->s, ni, AD_F_GRAD);
        next->s[S_TOS] = grad;
        next->s[S_PC]=pc+1; break;
    }
    /* All remaining opcodes delegate to exec loop via IS_NATIVE */
    case OP_NATIVE_CALL:
        next->s[S_IS_NATIVE]=1; next->s[S_PC]=pc+1; break;
    default:        next->s[S_PC]=pc+1; break;
    }
}

/* ── AD Backward Step: process one tape node per VM cycle ──
 * When AD_IS_BACKWARD is set, this function processes the node at AD_CURSOR,
 * propagates gradients to parent nodes, and decrements the cursor.
 * When cursor goes below 0, the backward pass is complete. */
/**
 * @brief Process one reverse-mode AD tape node per VM cycle when a backward
 *        pass is active (S_AD_IS_BACKWARD set): applies the gradient
 *        propagation rule for the node's recorded op (ADD/SUB/MUL/DIV/POW,
 *        or the generic dL = grad*saved rule for unary ops) to its parent
 *        node(s), accumulating into their AD_F_GRAD fields, then decrements
 *        the tape cursor. Products use the polarization identity
 *        (POLARIZATION_PRODUCT) rather than direct multiplication so the
 *        reference interpreter stays bit-identical to the matrix forward
 *        path's SQUARE-activation Layer 2. No-op when the cursor underflows
 *        past 0 (backward pass complete) or backward mode isn't active.
 */
static void ad_backward_step(float* s) {
    if (s[S_AD_IS_BACKWARD] < 0.5f) return;

    int cursor = (int)s[S_AD_CURSOR];
    if (cursor < 0) {
        /* Backward complete */
        s[S_AD_IS_BACKWARD] = 0;
        s[S_AD_MODE] = 0;
        return;
    }

    float grad = AD_NODE(s, cursor, AD_F_GRAD);
    float op_type = AD_NODE(s, cursor, AD_F_OP);
    float saved = AD_NODE(s, cursor, AD_F_SAVED);
    int li = (int)AD_NODE(s, cursor, AD_F_LEFT);
    int ri = (int)AD_NODE(s, cursor, AD_F_RIGHT);

    /* Gradient propagation rules.
     * Binary ops: ADD (dL=grad, dR=grad), SUB (dL=grad, dR=-grad),
     *             MUL (dL=grad*R_value, dR=grad*L_value),
     *             DIV (dL=grad/R, dR=-grad*L/(R*R)),
     *             POW (dL=grad*R*L^(R-1), dR=grad*L^R*log(L)).
     * Unary ops:  ALL use dL = grad * saved_val.
     *             saved_val is precomputed during forward recording:
     *             NEG=-1, ABS=sign, RELU=step, SIGMOID=val*(1-val),
     *             TANH=1-val², EXP=val, LOG=1/input, SQRT=1/(2*val),
     *             SIN=cos(input), COS=-sin(input). */
    if (fabsf(grad) > 1e-15f) {
        /* Cross products use the polarization identity
         *     a · b = ½·(a + b)² − ½·a² − ½·b²
         * which is what the matrix forward path's SQUARE FFN naturally
         * computes (Layer 2). For bit-identity with the matrix, the
         * reference VM must use the same arithmetic order — direct
         * multiplication and polarization are mathematically equal but
         * differ by 1–13 ULPs in float32. Without this change, sigmoid
         * (and any other unary AD op whose backward uses grad·saved)
         * shows a low-ULP divergence on the gradient output. */
        #define POLARIZATION_PRODUCT(a, b) \
            (0.5f * ((a) + (b)) * ((a) + (b)) - 0.5f * (a) * (a) - 0.5f * (b) * (b))
        if (fabsf(op_type - AD_OP_ADD) < 0.5f) {
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += grad;
            if (ri >= 0) AD_NODE(s, ri, AD_F_GRAD) += grad;
        } else if (fabsf(op_type - AD_OP_SUB) < 0.5f) {
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += grad;
            if (ri >= 0) AD_NODE(s, ri, AD_F_GRAD) -= grad;
        } else if (fabsf(op_type - AD_OP_MUL) < 0.5f) {
            float lv = (li >= 0) ? AD_NODE(s, li, AD_F_VALUE) : 0;
            float rv = (ri >= 0) ? AD_NODE(s, ri, AD_F_VALUE) : 0;
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += POLARIZATION_PRODUCT(grad, rv);
            if (ri >= 0) AD_NODE(s, ri, AD_F_GRAD) += POLARIZATION_PRODUCT(grad, lv);
        } else if (fabsf(op_type - AD_OP_DIV) < 0.5f) {
            float lv = (li >= 0) ? AD_NODE(s, li, AD_F_VALUE) : 0;
            float rv = (ri >= 0) ? AD_NODE(s, ri, AD_F_VALUE) : 0;
            float safe_rv = fabsf(rv) > 1e-15f ? rv : (rv < 0 ? -1e-15f : 1e-15f);
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += POLARIZATION_PRODUCT(grad, saved);
            if (ri >= 0) AD_NODE(s, ri, AD_F_GRAD) += POLARIZATION_PRODUCT(grad, lv) * (-1.0f / (safe_rv * safe_rv));
        } else if (fabsf(op_type - AD_OP_POW) < 0.5f) {
            float lv = (li >= 0) ? AD_NODE(s, li, AD_F_VALUE) : 0;
            float safe_lv = lv > 1e-15f ? lv : 1e-15f;
            float val = AD_NODE(s, cursor, AD_F_VALUE);
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += POLARIZATION_PRODUCT(grad, saved);
            if (ri >= 0) AD_NODE(s, ri, AD_F_GRAD) += grad * (val * logf(safe_lv));
        } else if ((op_type >= AD_OP_NEG && op_type <= AD_OP_SQRT + 0.5f) ||
                   fabsf(op_type - AD_OP_SIN) < 0.5f ||
                   fabsf(op_type - AD_OP_COS) < 0.5f) {
            /* ALL unary ops: dL = grad * saved_val */
            if (li >= 0) AD_NODE(s, li, AD_F_GRAD) += POLARIZATION_PRODUCT(grad, saved);
        }
        /* AD_OP_CONST and AD_OP_VAR: leaf nodes, no propagation */
        #undef POLARIZATION_PRODUCT
    }

    /* Decrement cursor */
    s[S_AD_CURSOR] = (float)(cursor - 1);
    if (cursor - 1 < 0) {
        s[S_AD_IS_BACKWARD] = 0;
        s[S_AD_MODE] = 0;
    }
}

static int g_last_ref_steps = 0;
static int g_last_sim_steps = 0;
static int g_last_mat_steps = 0;

/*******************************************************************************
 * Per-step JSONL trace emission
 *
 * Paper artifact: artifacts/paper/outputs/{vm,transformer}-traces.jsonl
 * Activated by main()'s --trace-vm <path> / --trace-transformer <path> flags.
 * The trace files are consumed by scripts/paper/compare_traces.py for the
 * fieldwise three-way agreement report (paper §4.4).
 *
 * Schema per line (matches scripts/paper/compare_traces.py):
 *   {"program":"<name>","step":<int>,
 *    "pc":<int>,"sp":<int>,"tos":<float>,"sos":<float>,
 *    "opcode":<int>,"is_native":<bool>,
 *    "registers":[r2,r3,depth],
 *    "memory":[mem0..mem3],
 *    "tape":[<8 tape values>],
 *    "flags":{"zero":<bool>,"halt":<bool>,"has_out":<bool>}}
 ******************************************************************************/
static FILE* g_trace_vm_fp = NULL;          /* set by main() if --trace-vm given */
static FILE* g_trace_tf_fp = NULL;          /* set by main() if --trace-transformer given */
static FILE* g_trace_sim_fp = NULL;         /* set by main() if --trace-simulated given */
static const char* g_trace_program_name = NULL; /* set by test() before each program */
static int g_trace_program_seq = -1;        /* per-test counter; disambiguates duplicate names */

/* JSON-safe number printer: %g loses bit-identity for f32 round-trips, so we
 * emit float bit pattern via %.9g (sufficient for IEEE 754 single precision).
 * Special-cases NaN/+inf/-inf as strings since JSON does not allow them as
 * bare tokens — paper claim is bitwise agreement, so the trace must round-trip. */
/** @brief Print a float to a JSONL trace file as a bit-identical %.9g
 *         number, with NaN/+-Infinity emitted as JSON strings since bare
 *         non-finite tokens aren't valid JSON. */
static void trace_emit_num(FILE* fp, float v) {
    if (v != v) { fputs("\"NaN\"", fp); return; }
    if (v >  3.4e38f) { fputs("\"Infinity\"", fp); return; }
    if (v < -3.4e38f) { fputs("\"-Infinity\"", fp); return; }
    fprintf(fp, "%.9g", (double)v);
}

/* is_native_override: pass >=0 to record the IS_NATIVE flag observed BEFORE
 * exec_loop_postprocess cleared it. Pass -1 to read s[S_IS_NATIVE] directly. */
/**
 * @brief Emit one JSONL trace record for the paper's fieldwise VM/transformer
 *        agreement comparison (program/pc/sp/tos/sos/output/opcode/
 *        is_native/registers/memory/tape/flags), if @p fp is non-null and a
 *        trace program name is set. @p is_native_override lets the caller
 *        record the IS_NATIVE flag observed before exec_loop_postprocess()
 *        clears it (pass -1 to read it directly from state).
 */
static void emit_trace_line(FILE* fp, int step, const float* s,
                            const Instr* prog, int n_instr,
                            int is_native_override) {
    if (!fp || !g_trace_program_name) return;
    int pc = (int)s[S_PC];
    int opcode = -1;
    if (pc >= 0 && pc < n_instr) opcode = (int)prog[pc].op;
    int is_native = is_native_override >= 0
                  ? (is_native_override > 0 ? 1 : 0)
                  : (s[S_IS_NATIVE] > 0.5f ? 1 : 0);
    int halt = s[S_HALT] > 0.5f ? 1 : 0;
    int has_out = s[S_HAS_OUT] > 0.5f ? 1 : 0;
    int zero = (s[S_TOS] == 0.0f) ? 1 : 0;

    fprintf(fp, "{\"program\":\"%s\",\"program_id\":%d,\"step\":%d,",
            g_trace_program_name, g_trace_program_seq, step);
    fprintf(fp, "\"pc\":%d,\"sp\":%d,", pc, (int)s[S_SP]);
    fputs("\"tos\":", fp);  trace_emit_num(fp, s[S_TOS]);
    fputs(",\"sos\":", fp); trace_emit_num(fp, s[S_SOS]);
    fputs(",\"output\":", fp); trace_emit_num(fp, s[S_OUTPUT]);
    fprintf(fp, ",\"opcode\":%d,\"is_native\":%s,", opcode, is_native ? "true" : "false");
    fputs("\"registers\":[", fp);
    trace_emit_num(fp, s[S_R2]); fputc(',', fp);
    trace_emit_num(fp, s[S_R3]); fputc(',', fp);
    trace_emit_num(fp, s[S_DEPTH]);
    fputs("],\"memory\":[", fp);
    trace_emit_num(fp, s[S_MEM0]); fputc(',', fp);
    trace_emit_num(fp, s[S_MEM1]); fputc(',', fp);
    trace_emit_num(fp, s[S_MEM2]); fputc(',', fp);
    trace_emit_num(fp, s[S_MEM3]);
    fputs("],\"tape\":[", fp);
    for (int i = 0; i < AD_MAX_TAPE; i++) {
        if (i > 0) fputc(',', fp);
        trace_emit_num(fp, s[S_AD_TAPE_BASE + i * AD_NODE_FIELDS + AD_F_VALUE]);
    }
    fprintf(fp, "],\"flags\":{\"zero\":%s,\"halt\":%s,\"has_out\":%s}}\n",
            zero ? "true" : "false",
            halt ? "true" : "false",
            has_out ? "true" : "false");
}

/**
 * @brief Run a program to completion (or a step cap) on the ground-truth
 *        reference interpreter (execute_step() / ad_backward_step()),
 *        resetting all global VM state (frames, heap, exception handlers,
 *        arena) first. Collects up to @p max_out output values and, if
 *        g_trace_vm_fp is set, emits a JSONL trace line per step.
 * @return The number of outputs produced.
 */
static int run_reference(const Instr* prog, int n_instr, float* outputs, int max_out) {
    /* Double-buffer instead of 8192-entry trace (saves ~1.15 MB stack) */
    State cur, nxt;
    state_init(&cur);
    g_frame_count = 0; g_heap_ptr = 0; g_exc_count = 0; g_current_exn = 0.0f; g_current_closure_ptr = -1; g_wind_depth = 0;
    if (g_vm_regions_initialized) { vm_arena_reset(&g_vm_regions.global_arena); }
    int n_out = 0, step_count = 0;
    /* Emit pre-step trace at step=0 for symmetry with the matrix runner. */
    emit_trace_line(g_trace_vm_fp, 0, cur.s, prog, n_instr, -1);
    while (step_count < 8191 && cur.s[S_HALT] < 0.5f) {
        step_count++;
        int pc = (int)cur.s[S_PC];
        if (pc >= 0 && pc < n_instr) {
            cur.s[S_OPCODE] = (float)prog[pc].op;
            cur.s[S_OPERAND] = (float)prog[pc].operand;
        }
        int is_native_pre = 0;
        /* If backward pass is active, process one tape node instead of a normal instruction */
        if (cur.s[S_AD_IS_BACKWARD] > 0.5f) {
            memcpy(&nxt, &cur, sizeof(State));
            ad_backward_step(nxt.s);
        } else {
            execute_step(&cur, prog, n_instr, &nxt);
            /* Capture IS_NATIVE before postprocess clears it. */
            is_native_pre = nxt.s[S_IS_NATIVE] > 0.5f ? 1 : 0;
            exec_loop_postprocess(nxt.s, prog, n_instr);
        }
        if (nxt.s[S_HAS_OUT] > 0.5f && n_out < max_out)
            outputs[n_out++] = nxt.s[S_OUTPUT];
        memcpy(&cur, &nxt, sizeof(State));
        emit_trace_line(g_trace_vm_fp, step_count, cur.s, prog, n_instr, is_native_pre);
    }
    g_last_ref_steps = step_count;
    return n_out;
}

/*******************************************************************************
 * 2. Simulated Transformer (C functions, verified correct)
 ******************************************************************************/

/**
 * @brief Simulated Layer 0 (instruction fetch): a Gaussian-peaked softmax
 *        attention over the @p np position embeddings @p pe, scored so the
 *        weight is sharply peaked at position == current PC (x[S_PC]),
 *        yielding the fetched opcode/operand as a weighted sum.
 */
