pub mod bytecode;
pub mod codegen;
pub use bytecode::{Instruction, Bytecode, Constant};
pub use codegen::CodeGenerator;
