use std::fmt;
use crate::types::OmegaType;

#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // Stack operations
    Push(Constant),
    Pop,
    Dup,
    Swap,
    Rot3,

    // Load/Store
    LoadLocal(u16),
    StoreLocal(u16),
    LoadGlobal(u16),
    StoreGlobal(u16),
    LoadUpvalue(u16),
    SetUpvalue(u16),
    LoadField(u16),
    StoreField(u16),
    LoadIndex,
    StoreIndex,

    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    FloorDiv,
    Neg,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,
    Not,

    // Control flow
    Jump(u32),
    JumpIfTrue(u32),
    JumpIfFalse(u32),
    JumpIfNone(u32),
    JumpBack(u32),

    // Function calls
    Call(u16),
    TailCall(u16),
    Return,
    Yield,

    // Closures
    MakeClosure(u16, Vec<u16>),
    MakeClass(u16),

    // Data structures
    MakeArray(u16),
    MakeMap(u16),
    MakeSet(u16),
    MakeTuple(u16),
    MakeStruct(u16),
    MakeEnum(u16),
    MakeRange(bool),

    // String operations
    FormatString(u16),
    StringConcat,

    // Iteration
    GetIter,
    IterNext,
    IterHasMore,

    // Pattern matching
    MatchPattern(u16),
    MatchGuard(u16),

    // Error handling
    PushCatch(u32),
    PopCatch,
    Throw,
    Rethrow,

    // Async/Await
    AsyncStart,
    AsyncEnd,
    Await,

    // Type operations
    CastType(u16),
    CheckType(u16),
    TypeOf,

    // Special
    Nop,
    Breakpoint,
    Halt,
    Assert,
    Print(bool),
    Defer(u32),
    Drop,
    RefCount(u16),
    IncRef,
    DecRef,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    None,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Char(char),
    Byte(u8),
    BigInt(String),
    Function(FunctionConstant),
    Struct(StructConstant),
    Enum(EnumConstant),
    Type(OmegaType),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionConstant {
    pub name: String,
    pub arity: u16,
    pub upvalue_count: u16,
    pub chunk_index: usize,
    pub is_async: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructConstant {
    pub name: String,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumConstant {
    pub name: String,
    pub variants: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Bytecode {
    pub instructions: Vec<Instruction>,
    pub constants: Vec<Constant>,
    pub lines: Vec<u32>,
    pub name: String,
    pub arity: u16,
    pub local_count: u16,
    pub upvalue_count: u16,
    pub is_async: bool,
}

impl Bytecode {
    pub fn new(name: String) -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
            lines: Vec::new(),
            name,
            arity: 0,
            local_count: 0,
            upvalue_count: 0,
            is_async: false,
        }
    }

    pub fn emit(&mut self, instruction: Instruction, line: u32) -> usize {
        let index = self.instructions.len();
        self.instructions.push(instruction);
        self.lines.push(line);
        index
    }

    pub fn emit_jump(&mut self, instruction: Instruction, line: u32) -> usize {
        let index = self.emit(instruction, line);
        index
    }

    pub fn patch_jump(&mut self, jump_index: usize) {
        let target = self.instructions.len() as u32;
        match &mut self.instructions[jump_index] {
            Instruction::Jump(ref mut t) |
            Instruction::JumpIfTrue(ref mut t) |
            Instruction::JumpIfFalse(ref mut t) |
            Instruction::JumpIfNone(ref mut t) => {
                *t = target;
            }
            _ => {}
        }
    }

    pub fn add_constant(&mut self, constant: Constant) -> u16 {
        let index = self.constants.len();
        self.constants.push(constant);
        index as u16
    }

    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    pub fn disassemble(&self) -> String {
        let mut output = format!("=== {} ===\n", self.name);
        for (i, instruction) in self.instructions.iter().enumerate() {
            let line = self.lines.get(i).unwrap_or(&0);
            output.push_str(&format!("{:04} {:4} | {:?}\n", i, line, instruction));
        }
        output.push_str("\nConstants:\n");
        for (i, constant) in self.constants.iter().enumerate() {
            output.push_str(&format!("  {:04} | {:?}\n", i, constant));
        }
        output
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Push(c) => write!(f, "push {:?}", c),
            Instruction::Pop => write!(f, "pop"),
            Instruction::Dup => write!(f, "dup"),
            Instruction::Swap => write!(f, "swap"),
            Instruction::Rot3 => write!(f, "rot3"),
            Instruction::LoadLocal(i) => write!(f, "load_local {}", i),
            Instruction::StoreLocal(i) => write!(f, "store_local {}", i),
            Instruction::LoadGlobal(i) => write!(f, "load_global {}", i),
            Instruction::StoreGlobal(i) => write!(f, "store_global {}", i),
            Instruction::LoadUpvalue(i) => write!(f, "load_upvalue {}", i),
            Instruction::SetUpvalue(i) => write!(f, "setupvalue {}", i),
            Instruction::LoadField(i) => write!(f, "load_field {}", i),
            Instruction::StoreField(i) => write!(f, "store_field {}", i),
            Instruction::LoadIndex => write!(f, "load_index"),
            Instruction::StoreIndex => write!(f, "store_index"),
            Instruction::Add => write!(f, "add"),
            Instruction::Sub => write!(f, "sub"),
            Instruction::Mul => write!(f, "mul"),
            Instruction::Div => write!(f, "div"),
            Instruction::Mod => write!(f, "mod"),
            Instruction::Pow => write!(f, "pow"),
            Instruction::FloorDiv => write!(f, "floor_div"),
            Instruction::Neg => write!(f, "neg"),
            Instruction::BitAnd => write!(f, "bit_and"),
            Instruction::BitOr => write!(f, "bit_or"),
            Instruction::BitXor => write!(f, "bit_xor"),
            Instruction::BitNot => write!(f, "bit_not"),
            Instruction::Shl => write!(f, "shl"),
            Instruction::Shr => write!(f, "shr"),
            Instruction::Eq => write!(f, "eq"),
            Instruction::Ne => write!(f, "ne"),
            Instruction::Lt => write!(f, "lt"),
            Instruction::Le => write!(f, "le"),
            Instruction::Gt => write!(f, "gt"),
            Instruction::Ge => write!(f, "ge"),
            Instruction::And => write!(f, "and"),
            Instruction::Or => write!(f, "or"),
            Instruction::Not => write!(f, "not"),
            Instruction::Jump(t) => write!(f, "jump {}", t),
            Instruction::JumpIfTrue(t) => write!(f, "jump_if_true {}", t),
            Instruction::JumpIfFalse(t) => write!(f, "jump_if_false {}", t),
            Instruction::JumpIfNone(t) => write!(f, "jump_if_none {}", t),
            Instruction::JumpBack(t) => write!(f, "jump_back {}", t),
            Instruction::Call(a) => write!(f, "call {}", a),
            Instruction::TailCall(a) => write!(f, "tail_call {}", a),
            Instruction::Return => write!(f, "return"),
            Instruction::Yield => write!(f, "yield"),
            Instruction::MakeClosure(i, u) => write!(f, "make_closure {} {:?}", i, u),
            Instruction::MakeClass(i) => write!(f, "make_class {}", i),
            Instruction::MakeArray(n) => write!(f, "make_array {}", n),
            Instruction::MakeMap(n) => write!(f, "make_map {}", n),
            Instruction::MakeSet(n) => write!(f, "make_set {}", n),
            Instruction::MakeTuple(n) => write!(f, "make_tuple {}", n),
            Instruction::MakeStruct(n) => write!(f, "make_struct {}", n),
            Instruction::MakeEnum(n) => write!(f, "make_enum {}", n),
            Instruction::MakeRange(i) => write!(f, "make_range {}", i),
            Instruction::FormatString(n) => write!(f, "format_string {}", n),
            Instruction::StringConcat => write!(f, "string_concat"),
            Instruction::GetIter => write!(f, "get_iter"),
            Instruction::IterNext => write!(f, "iter_next"),
            Instruction::IterHasMore => write!(f, "iter_has_more"),
            Instruction::MatchPattern(i) => write!(f, "match_pattern {}", i),
            Instruction::MatchGuard(i) => write!(f, "match_guard {}", i),
            Instruction::PushCatch(t) => write!(f, "push_catch {}", t),
            Instruction::PopCatch => write!(f, "pop_catch"),
            Instruction::Throw => write!(f, "throw"),
            Instruction::Rethrow => write!(f, "rethrow"),
            Instruction::AsyncStart => write!(f, "async_start"),
            Instruction::AsyncEnd => write!(f, "async_end"),
            Instruction::Await => write!(f, "await"),
            Instruction::CastType(i) => write!(f, "cast_type {}", i),
            Instruction::CheckType(i) => write!(f, "check_type {}", i),
            Instruction::TypeOf => write!(f, "typeof"),
            Instruction::Nop => write!(f, "nop"),
            Instruction::Breakpoint => write!(f, "breakpoint"),
            Instruction::Halt => write!(f, "halt"),
            Instruction::Assert => write!(f, "assert"),
            Instruction::Print(n) => write!(f, "print {}", n),
            Instruction::Defer(t) => write!(f, "defer {}", t),
            Instruction::Drop => write!(f, "drop"),
            Instruction::RefCount(i) => write!(f, "ref_count {}", i),
            Instruction::IncRef => write!(f, "inc_ref"),
            Instruction::DecRef => write!(f, "dec_ref"),
        }
    }
}
