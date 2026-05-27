use crate::ast::*;
use crate::errors::{OmegaError, OmegaResult};
use crate::types::OmegaType;
use super::bytecode::{Bytecode, Constant, Instruction, FunctionConstant, StructConstant, EnumConstant};

pub struct CodeGenerator {
    chunks: Vec<Bytecode>,
    current_chunk: usize,
    scope_depth: usize,
    locals: Vec<Local>,
    upvalues: Vec<Upvalue>,
    loop_stack: Vec<LoopContext>,
    defer_stack: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
struct Local {
    name: String,
    depth: usize,
    is_captured: bool,
    is_mutable: bool,
}

#[derive(Debug, Clone)]
struct Upvalue {
    index: u16,
    is_local: bool,
}

#[derive(Debug)]
struct LoopContext {
    break_jumps: Vec<usize>,
    continue_target: usize,
}

impl CodeGenerator {
    pub fn new() -> Self {
        let main_chunk = Bytecode::new("<main>".to_string());
        Self {
            chunks: vec![main_chunk],
            current_chunk: 0,
            scope_depth: 0,
            locals: Vec::new(),
            upvalues: Vec::new(),
            loop_stack: Vec::new(),
            defer_stack: Vec::new(),
        }
    }

    pub fn compile(&mut self, ast: &AstNode) -> OmegaResult<()> {
        self.compile_node(ast)?;
        self.emit(Instruction::Halt);
        Ok(())
    }

    pub fn get_chunks(&self) -> &[Bytecode] {
        &self.chunks
    }

    pub fn get_main_chunk(&self) -> &Bytecode {
        &self.chunks[0]
    }

    fn current(&self) -> &Bytecode {
        &self.chunks[self.current_chunk]
    }

    fn current_mut(&mut self) -> &mut Bytecode {
        &mut self.chunks[self.current_chunk]
    }

    fn emit(&mut self, instruction: Instruction) -> usize {
        let line = 0; // TODO: track line numbers
        self.current_mut().emit(instruction, line)
    }

    fn emit_jump(&mut self, instruction: Instruction) -> usize {
        self.current_mut().emit_jump(instruction, 0)
    }

    fn patch_jump(&mut self, index: usize) {
        self.current_mut().patch_jump(index);
    }

    fn add_constant(&mut self, constant: Constant) -> u16 {
        self.current_mut().add_constant(constant)
    }

    fn compile_node(&mut self, node: &AstNode) -> OmegaResult<()> {
        match node {
            AstNode::Program(stmts) => {
                for stmt in stmts {
                    self.compile_node(stmt)?;
                }
            }
            AstNode::Block(stmts) => {
                self.begin_scope();
                for stmt in stmts {
                    self.compile_node(stmt)?;
                }
                self.end_scope();
            }
            AstNode::IntegerLiteral(v) => {
                let idx = self.add_constant(Constant::Integer(*v));
                self.emit(Instruction::Push(Constant::Integer(*v)));
            }
            AstNode::FloatLiteral(v) => {
                self.emit(Instruction::Push(Constant::Float(*v)));
            }
            AstNode::StringLiteral(v) => {
                self.emit(Instruction::Push(Constant::String(v.clone())));
            }
            AstNode::BoolLiteral(v) => {
                self.emit(Instruction::Push(Constant::Bool(*v)));
            }
            AstNode::CharLiteral(v) => {
                self.emit(Instruction::Push(Constant::Char(*v)));
            }
            AstNode::NoneLiteral => {
                self.emit(Instruction::Push(Constant::None));
            }
            AstNode::BigIntLiteral(v) => {
                self.emit(Instruction::Push(Constant::BigInt(v.clone())));
            }
            AstNode::Identifier(name) => {
                self.compile_load_variable(name)?;
            }
            AstNode::BinaryOp { op, left, right } => {
                self.compile_node(left)?;
                self.compile_node(right)?;
                match op {
                    BinaryOp::Add => self.emit(Instruction::Add),
                    BinaryOp::Sub => self.emit(Instruction::Sub),
                    BinaryOp::Mul => self.emit(Instruction::Mul),
                    BinaryOp::Div => self.emit(Instruction::Div),
                    BinaryOp::Mod => self.emit(Instruction::Mod),
                    BinaryOp::Pow => self.emit(Instruction::Pow),
                    BinaryOp::FloorDiv => self.emit(Instruction::FloorDiv),
                    BinaryOp::BitAnd => self.emit(Instruction::BitAnd),
                    BinaryOp::BitOr => self.emit(Instruction::BitOr),
                    BinaryOp::BitXor => self.emit(Instruction::BitXor),
                    BinaryOp::Shl => self.emit(Instruction::Shl),
                    BinaryOp::Shr => self.emit(Instruction::Shr),
                    BinaryOp::Eq => self.emit(Instruction::Eq),
                    BinaryOp::Ne => self.emit(Instruction::Ne),
                    BinaryOp::Lt => self.emit(Instruction::Lt),
                    BinaryOp::Le => self.emit(Instruction::Le),
                    BinaryOp::Gt => self.emit(Instruction::Gt),
                    BinaryOp::Ge => self.emit(Instruction::Ge),
                    BinaryOp::And => {
                        let jump = self.emit_jump(Instruction::JumpIfFalse(0));
                        self.emit(Instruction::Pop);
                        self.compile_node(right)?;
                        self.patch_jump(jump);
                        return Ok(());
                    }
                    BinaryOp::Or => {
                        let jump = self.emit_jump(Instruction::JumpIfTrue(0));
                        self.emit(Instruction::Pop);
                        self.compile_node(right)?;
                        self.patch_jump(jump);
                        return Ok(());
                    }
                    _ => self.emit(Instruction::Nop),
                };
            }
            AstNode::UnaryOp { op, operand } => {
                self.compile_node(operand)?;
                match op {
                    UnaryOp::Neg => { self.emit(Instruction::Neg); }
                    UnaryOp::Not => { self.emit(Instruction::Not); }
                    UnaryOp::BitNot => { self.emit(Instruction::BitNot); }
                    _ => {}
                }
            }
            AstNode::LetBinding { name, value, .. } => {
                if let Some(val) = value {
                    self.compile_node(val)?;
                } else {
                    self.emit(Instruction::Push(Constant::None));
                }
                let local = self.add_local(name.clone(), false);
                self.emit(Instruction::StoreLocal(local));
            }
            AstNode::Assign { target, value } => {
                self.compile_node(value)?;
                match target.as_ref() {
                    AstNode::Identifier(name) => {
                        self.compile_store_variable(name)?;
                    }
                    AstNode::Index { object, index } => {
                        self.compile_node(object)?;
                        self.compile_node(index)?;
                        self.emit(Instruction::StoreIndex);
                    }
                    AstNode::Attribute { object, attribute } => {
                        self.compile_node(object)?;
                        let idx = self.add_constant(Constant::String(attribute.clone()));
                        self.emit(Instruction::StoreField(idx));
                    }
                    _ => return Err(OmegaError::CompilationError {
                        message: "Invalid assignment target".to_string(),
                        span: None,
                    }),
                }
            }
            AstNode::CompoundAssign { op, target, value } => {
                match target.as_ref() {
                    AstNode::Identifier(name) => {
                        self.compile_load_variable(name)?;
                        self.compile_node(value)?;
                        match op {
                            BinaryOp::Add => { self.emit(Instruction::Add); }
                            BinaryOp::Sub => { self.emit(Instruction::Sub); }
                            BinaryOp::Mul => { self.emit(Instruction::Mul); }
                            BinaryOp::Div => { self.emit(Instruction::Div); }
                            BinaryOp::Mod => { self.emit(Instruction::Mod); }
                            _ => {}
                        }
                        self.compile_store_variable(name)?;
                    }
                    _ => return Err(OmegaError::CompilationError {
                        message: "Invalid compound assignment target".to_string(),
                        span: None,
                    }),
                }
            }
            AstNode::Call { function, args, .. } => {
                self.compile_node(function)?;
                for arg in args {
                    self.compile_node(arg)?;
                }
                self.emit(Instruction::Call(args.len() as u16));
            }
            AstNode::MethodCall { object, method, args, .. } => {
                self.compile_node(object)?;
                let method_idx = self.add_constant(Constant::String(method.clone()));
                self.emit(Instruction::LoadField(method_idx));
                self.compile_node(object)?; // push self
                for arg in args {
                    self.compile_node(arg)?;
                }
                self.emit(Instruction::Call((args.len() + 1) as u16));
            }
            AstNode::If { condition, then_branch, elif_branches, else_branch } => {
                self.compile_node(condition)?;
                let then_jump = self.emit_jump(Instruction::JumpIfFalse(0));

                self.compile_node(then_branch)?;
                let else_jump = self.emit_jump(Instruction::Jump(0));

                self.patch_jump(then_jump);
                self.emit(Instruction::Pop);

                for (elif_cond, elif_body) in elif_branches {
                    self.compile_node(elif_cond)?;
                    let elif_jump = self.emit_jump(Instruction::JumpIfFalse(0));
                    self.compile_node(elif_body)?;
                    let elif_end = self.emit_jump(Instruction::Jump(0));
                    self.patch_jump(elif_jump);
                    self.emit(Instruction::Pop);
                }

                if let Some(else_b) = else_branch {
                    self.compile_node(else_b)?;
                }

                self.patch_jump(else_jump);
            }
            AstNode::While { condition, body } => {
                let loop_start = self.current().instructions.len();
                self.loop_stack.push(LoopContext {
                    break_jumps: Vec::new(),
                    continue_target: loop_start,
                });

                self.compile_node(condition)?;
                let exit_jump = self.emit_jump(Instruction::JumpIfFalse(0));
                self.emit(Instruction::Pop);

                self.compile_node(body)?;
                self.emit(Instruction::JumpBack(loop_start as u32));

                self.patch_jump(exit_jump);
                self.emit(Instruction::Pop);

                let loop_ctx = self.loop_stack.pop().unwrap();
                for jump in loop_ctx.break_jumps {
                    self.patch_jump(jump);
                }
            }
            AstNode::For { variable, iterable, body } => {
                self.compile_node(iterable)?;
                self.emit(Instruction::GetIter);

                let loop_start = self.current().instructions.len();
                self.loop_stack.push(LoopContext {
                    break_jumps: Vec::new(),
                    continue_target: loop_start,
                });

                let has_more_jump = self.emit_jump(Instruction::IterHasMore);
                self.emit(Instruction::IterNext);
                let local = self.add_local(variable.clone(), true);
                self.emit(Instruction::StoreLocal(local));

                self.compile_node(body)?;
                self.emit(Instruction::JumpBack(loop_start as u32));

                self.patch_jump(has_more_jump);

                let loop_ctx = self.loop_stack.pop().unwrap();
                for jump in loop_ctx.break_jumps {
                    self.patch_jump(jump);
                }
            }
            AstNode::Loop { body } => {
                let loop_start = self.current().instructions.len();
                self.loop_stack.push(LoopContext {
                    break_jumps: Vec::new(),
                    continue_target: loop_start,
                });

                self.compile_node(body)?;
                self.emit(Instruction::JumpBack(loop_start as u32));

                let loop_ctx = self.loop_stack.pop().unwrap();
                for jump in loop_ctx.break_jumps {
                    self.patch_jump(jump);
                }
            }
            AstNode::Break { .. } => {
                let jump = self.emit_jump(Instruction::Jump(0));
                if let Some(loop_ctx) = self.loop_stack.last_mut() {
                    loop_ctx.break_jumps.push(jump);
                }
            }
            AstNode::Continue => {
                if let Some(loop_ctx) = self.loop_stack.last() {
                    self.emit(Instruction::JumpBack(loop_ctx.continue_target as u32));
                }
            }
            AstNode::Return { value } => {
                self.emit_defers();
                if let Some(v) = value {
                    self.compile_node(v)?;
                } else {
                    self.emit(Instruction::Push(Constant::None));
                }
                self.emit(Instruction::Return);
            }
            AstNode::FunctionDef { name, params, body, is_async, .. } => {
                let chunk_index = self.chunks.len();
                let mut chunk = Bytecode::new(name.clone());
                chunk.arity = params.len() as u16;
                chunk.is_async = *is_async;
                self.chunks.push(chunk);

                let old_chunk = self.current_chunk;
                let old_locals = self.locals.clone();
                let old_upvalues = self.upvalues.clone();
                self.current_chunk = chunk_index;
                self.locals = Vec::new();
                self.upvalues = Vec::new();

                for param in params {
                    self.add_local(param.name.clone(), true);
                }

                self.compile_node(body)?;
                self.emit(Instruction::Push(Constant::None));
                self.emit(Instruction::Return);

                let upvalue_count = self.upvalues.len() as u16;
                self.chunks[chunk_index].upvalue_count = upvalue_count;

                self.current_chunk = old_chunk;
                self.locals = old_locals;
                self.upvalues = old_upvalues;

                let func_const = Constant::Function(FunctionConstant {
                    name: name.clone(),
                    arity: params.len() as u16,
                    upvalue_count,
                    chunk_index,
                    is_async: *is_async,
                });
                let idx = self.add_constant(func_const);

                if upvalue_count > 0 {
                    let upvalue_indices: Vec<u16> = (0..upvalue_count).collect();
                    self.emit(Instruction::MakeClosure(idx, upvalue_indices));
                } else {
                    self.emit(Instruction::Push(func_const));
                }

                let local = self.add_local(name.clone(), false);
                self.emit(Instruction::StoreLocal(local));
            }
            AstNode::Array(elements) => {
                let count = elements.len();
                for elem in elements {
                    self.compile_node(elem)?;
                }
                self.emit(Instruction::MakeArray(count as u16));
            }
            AstNode::Map(entries) => {
                let count = entries.len();
                for (k, v) in entries {
                    self.compile_node(k)?;
                    self.compile_node(v)?;
                }
                self.emit(Instruction::MakeMap(count as u16));
            }
            AstNode::Tuple(elements) => {
                let count = elements.len();
                for elem in elements {
                    self.compile_node(elem)?;
                }
                self.emit(Instruction::MakeTuple(count as u16));
            }
            AstNode::Range { start, end, inclusive } => {
                self.compile_node(start)?;
                self.compile_node(end)?;
                self.emit(Instruction::MakeRange(*inclusive));
            }
            AstNode::Index { object, index } => {
                self.compile_node(object)?;
                self.compile_node(index)?;
                self.emit(Instruction::LoadIndex);
            }
            AstNode::Attribute { object, attribute } => {
                self.compile_node(object)?;
                let idx = self.add_constant(Constant::String(attribute.clone()));
                self.emit(Instruction::LoadField(idx));
            }
            AstNode::Print { args, newline } => {
                for arg in args {
                    self.compile_node(arg)?;
                }
                self.emit(Instruction::Print(*newline));
            }
            AstNode::Assert { condition, message } => {
                self.compile_node(condition)?;
                if let Some(msg) = message {
                    self.compile_node(msg)?;
                } else {
                    self.emit(Instruction::Push(Constant::String("Assertion failed".to_string())));
                }
                self.emit(Instruction::Assert);
            }
            AstNode::Throw { value } => {
                self.compile_node(value)?;
                self.emit(Instruction::Throw);
            }
            AstNode::TryCatch { try_body, catch_clauses, finally_body } => {
                let catch_jump = self.emit_jump(Instruction::PushCatch(0));
                self.compile_node(try_body)?;
                self.emit(Instruction::PopCatch);

                let finally_jump = self.emit_jump(Instruction::Jump(0));
                self.patch_jump(catch_jump);

                for clause in catch_clauses {
                    if let Some(binding) = &clause.binding {
                        let local = self.add_local(binding.clone(), true);
                        self.emit(Instruction::StoreLocal(local));
                    } else {
                        self.emit(Instruction::Pop);
                    }
                    self.compile_node(&clause.body)?;
                }

                self.patch_jump(finally_jump);
                if let Some(finally_b) = finally_body {
                    self.compile_node(finally_b)?;
                }
            }
            AstNode::Defer { body } => {
                let defer_jump = self.emit_jump(Instruction::Jump(0));
                let defer_start = self.current().instructions.len();
                self.compile_node(body)?;
                self.emit(Instruction::Return);
                self.patch_jump(defer_jump);

                if self.defer_stack.is_empty() {
                    self.defer_stack.push(Vec::new());
                }
                self.defer_stack.last_mut().unwrap().push(defer_start);
            }
            AstNode::StructDef { name, fields, .. } => {
                let field_names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
                let struct_const = Constant::Struct(StructConstant {
                    name: name.clone(),
                    fields: field_names,
                });
                self.add_constant(struct_const);
            }
            AstNode::EnumDef { name, variants, .. } => {
                let variant_names: Vec<String> = variants.iter().map(|v| v.name.clone()).collect();
                let enum_const = Constant::Enum(EnumConstant {
                    name: name.clone(),
                    variants: variant_names,
                });
                self.add_constant(enum_const);
            }
            _ => {
                self.emit(Instruction::Nop);
            }
        }
        Ok(())
    }

    fn compile_load_variable(&mut self, name: &str) -> OmegaResult<()> {
        if let Some(local) = self.resolve_local(name) {
            self.emit(Instruction::LoadLocal(local));
        } else if let Some(upvalue) = self.resolve_upvalue(name) {
            self.emit(Instruction::LoadUpvalue(upvalue));
        } else {
            let idx = self.add_constant(Constant::String(name.to_string()));
            self.emit(Instruction::LoadGlobal(idx));
        }
        Ok(())
    }

    fn compile_store_variable(&mut self, name: &str) -> OmegaResult<()> {
        if let Some(local) = self.resolve_local(name) {
            self.emit(Instruction::StoreLocal(local));
        } else if let Some(upvalue) = self.resolve_upvalue(name) {
            self.emit(Instruction::SetUpvalue(upvalue));
        } else {
            let idx = self.add_constant(Constant::String(name.to_string()));
            self.emit(Instruction::StoreGlobal(idx));
        }
        Ok(())
    }

    fn resolve_local(&self, name: &str) -> Option<u16> {
        for (i, local) in self.locals.iter().enumerate().rev() {
            if local.name == name {
                return Some(i as u16);
            }
        }
        None
    }

    fn resolve_upvalue(&mut self, name: &str) -> Option<u16> {
        if self.scope_depth == 0 {
            return None;
        }

        // Check parent scope
        for (i, local) in self.locals.iter().enumerate().rev() {
            if local.name == name {
                // Mark as captured
                let idx = i as u16;
                self.upvalues.push(Upvalue { index: idx, is_local: true });
                return Some((self.upvalues.len() - 1) as u16);
            }
        }

        None
    }

    fn add_local(&mut self, name: String, is_mutable: bool) -> u16 {
        let index = self.locals.len();
        self.locals.push(Local {
            name,
            depth: self.scope_depth,
            is_captured: false,
            is_mutable,
        });
        index as u16
    }

    fn begin_scope(&mut self) {
        self.scope_depth += 1;
    }

    fn end_scope(&mut self) {
        self.scope_depth -= 1;

        while self.locals.last().map_or(false, |l| l.depth > self.scope_depth) {
            if self.locals.last().unwrap().is_captured {
                self.emit(Instruction::Drop);
            } else {
                self.emit(Instruction::Pop);
            }
            self.locals.pop();
        }
    }

    fn emit_defers(&mut self) {
        if let Some(defers) = self.defer_stack.last() {
            for &defer_start in defers.iter().rev() {
                self.emit(Instruction::Defer(defer_start as u32));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    fn compile(source: &str) -> OmegaResult<Bytecode> {
        let mut parser = Parser::new(source);
        let ast = parser.parse()?;
        let mut codegen = CodeGenerator::new();
        codegen.compile(&ast)?;
        Ok(codegen.get_main_chunk().clone())
    }

    #[test]
    fn test_compile_literals() {
        let bytecode = compile("42").unwrap();
        assert!(!bytecode.instructions.is_empty());
    }

    #[test]
    fn test_compile_binary_op() {
        let bytecode = compile("1 + 2").unwrap();
        assert!(bytecode.instructions.contains(&Instruction::Add));
    }

    #[test]
    fn test_compile_let() {
        let bytecode = compile("let x = 42").unwrap();
        assert!(bytecode.instructions.iter().any(|i| matches!(i, Instruction::StoreLocal(_))));
    }
}
