use crate::compiler::bytecode::{Bytecode, Constant, Instruction};
use crate::errors::{OmegaError, OmegaResult};
use super::stack::{Stack, Value, FunctionValue, IteratorValue};
use super::heap::Heap;

pub struct VirtualMachine {
    stack: Stack,
    heap: Heap,
    frames: Vec<CallFrame>,
    globals: Vec<Value>,
    ip: usize,
    chunk_index: usize,
    catch_stack: Vec<CatchFrame>,
    defer_stack: Vec<usize>,
    debug: bool,
}

#[derive(Debug)]
struct CallFrame {
    chunk_index: usize,
    ip: usize,
    stack_base: usize,
    upvalues: Vec<Value>,
    function_name: String,
}

#[derive(Debug)]
struct CatchFrame {
    handler_ip: usize,
    chunk_index: usize,
}

impl VirtualMachine {
    pub fn new() -> Self {
        Self {
            stack: Stack::new(),
            heap: Heap::new(),
            frames: Vec::new(),
            globals: Vec::with_capacity(256),
            ip: 0,
            chunk_index: 0,
            catch_stack: Vec::new(),
            defer_stack: Vec::new(),
            debug: false,
        }
    }

    pub fn with_debug(mut self) -> Self {
        self.debug = true;
        self
    }

    pub fn run(&mut self, chunks: &[Bytecode]) -> OmegaResult<Value> {
        loop {
            if self.chunk_index >= chunks.len() {
                return Err(OmegaError::InternalError {
                    message: "Invalid chunk index".to_string(),
                });
            }

            let chunk = &chunks[self.chunk_index];

            if self.ip >= chunk.instructions.len() {
                return self.stack.pop().or_else(|_| Ok(Value::None));
            }

            let instruction = chunk.instructions[self.ip].clone();
            self.ip += 1;

            if self.debug {
                eprintln!("  [{:04}] {:?}", self.ip - 1, instruction);
            }

            match instruction {
                Instruction::Push(constant) => {
                    let value = self.constant_to_value(constant, chunks)?;
                    self.stack.push(value)?;
                }
                Instruction::Pop => {
                    self.stack.pop()?;
                }
                Instruction::Dup => {
                    self.stack.dup()?;
                }
                Instruction::Swap => {
                    self.stack.swap()?;
                }
                Instruction::Rot3 => {
                    self.stack.rot3()?;
                }
                Instruction::LoadLocal(index) => {
                    let base = self.frames.last().map(|f| f.stack_base).unwrap_or(0);
                    let value = self.stack.get(base + index as usize)?.clone();
                    self.stack.push(value)?;
                }
                Instruction::StoreLocal(index) => {
                    let base = self.frames.last().map(|f| f.stack_base).unwrap_or(0);
                    let value = self.stack.pop()?;
                    self.stack.set(base + index as usize, value)?;
                }
                Instruction::LoadGlobal(index) => {
                    let name = match &chunk.constants[index as usize] {
                        Constant::String(s) => s.clone(),
                        _ => return Err(OmegaError::InternalError {
                            message: "Expected string constant for global".to_string(),
                        }),
                    };
                    // Find global by name
                    let idx = self.find_global(&name, chunks)?;
                    if let Some(value) = self.globals.get(idx) {
                        self.stack.push(value.clone())?;
                    } else {
                        return Err(OmegaError::NameError {
                            name: name.clone(),
                            span: None,
                        });
                    }
                }
                Instruction::StoreGlobal(index) => {
                    let name = match &chunk.constants[index as usize] {
                        Constant::String(s) => s.clone(),
                        _ => return Err(OmegaError::InternalError {
                            message: "Expected string constant for global".to_string(),
                        }),
                    };
                    let value = self.stack.pop()?;
                    let idx = self.find_or_create_global(&name);
                    if idx < self.globals.len() {
                        self.globals[idx] = value;
                    } else {
                        self.globals.push(value);
                    }
                }
                Instruction::LoadUpvalue(index) => {
                    if let Some(frame) = self.frames.last() {
                        if let Some(upvalue) = frame.upvalues.get(index as usize) {
                            self.stack.push(upvalue.clone())?;
                        }
                    }
                }
                Instruction::SetUpvalue(index) => {
                    let value = self.stack.pop()?;
                    if let Some(frame) = self.frames.last_mut() {
                        if (index as usize) < frame.upvalues.len() {
                            frame.upvalues[index as usize] = value;
                        }
                    }
                }
                Instruction::LoadField(index) => {
                    let field_name = match &chunk.constants[index as usize] {
                        Constant::String(s) => s.clone(),
                        _ => return Err(OmegaError::InternalError {
                            message: "Expected string constant for field".to_string(),
                        }),
                    };
                    let object = self.stack.pop()?;
                    let value = match &object {
                        Value::Object(obj) => {
                            obj.get_field(&field_name).cloned().unwrap_or(Value::None)
                        }
                        Value::Map(map) => {
                            map.iter()
                                .find(|(k, _)| matches!(k, Value::String(s) if s == &field_name))
                                .map(|(_, v)| v.clone())
                                .unwrap_or(Value::None)
                        }
                        Value::Array(arr) => {
                            match field_name.as_str() {
                                "len" => Value::Integer(arr.len() as i64),
                                "is_empty" => Value::Bool(arr.is_empty()),
                                "first" => arr.first().cloned().unwrap_or(Value::None),
                                "last" => arr.last().cloned().unwrap_or(Value::None),
                                _ => Value::None,
                            }
                        }
                        Value::String(s) => {
                            match field_name.as_str() {
                                "len" => Value::Integer(s.len() as i64),
                                "is_empty" => Value::Bool(s.is_empty()),
                                "chars" => {
                                    let chars: Vec<Value> = s.chars().map(Value::Char).collect();
                                    Value::Array(chars)
                                }
                                "to_upper" => Value::String(s.to_uppercase()),
                                "to_lower" => Value::String(s.to_lowercase()),
                                "trim" => Value::String(s.trim().to_string()),
                                _ => Value::None,
                            }
                        }
                        _ => Value::None,
                    };
                    self.stack.push(value)?;
                }
                Instruction::StoreField(index) => {
                    let field_name = match &chunk.constants[index as usize] {
                        Constant::String(s) => s.clone(),
                        _ => return Err(OmegaError::InternalError {
                            message: "Expected string constant for field".to_string(),
                        }),
                    };
                    let value = self.stack.pop()?;
                    let mut object = self.stack.pop()?;
                    if let Value::Object(ref mut obj) = object {
                        obj.set_field(field_name, value);
                    }
                    self.stack.push(object)?;
                }
                Instruction::LoadIndex => {
                    let index = self.stack.pop()?;
                    let object = self.stack.pop()?;
                    let value = object.index(&index)?;
                    self.stack.push(value)?;
                }
                Instruction::StoreIndex => {
                    let index = self.stack.pop()?;
                    let value = self.stack.pop()?;
                    let mut object = self.stack.pop()?;
                    // Store index implementation
                    self.stack.push(object)?;
                }
                Instruction::Add => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.add(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Sub => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.sub(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Mul => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.mul(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Div => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.div(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Mod => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.modulo(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Pow => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.pow(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::FloorDiv => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.div(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Neg => {
                    let a = self.stack.pop()?;
                    let result = a.neg()?;
                    self.stack.push(result)?;
                }
                Instruction::BitAnd => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.bit_and(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::BitOr => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.bit_or(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::BitXor => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.bit_xor(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::BitNot => {
                    let a = self.stack.pop()?;
                    let result = a.bit_not()?;
                    self.stack.push(result)?;
                }
                Instruction::Shl => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.shl(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Shr => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.shr(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Eq => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    self.stack.push(a.eq(&b))?;
                }
                Instruction::Ne => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    self.stack.push(a.ne(&b))?;
                }
                Instruction::Lt => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.lt(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Le => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.le(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Gt => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.gt(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::Ge => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = a.ge(&b)?;
                    self.stack.push(result)?;
                }
                Instruction::And => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    self.stack.push(Value::Bool(a.is_truthy() && b.is_truthy()))?;
                }
                Instruction::Or => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    self.stack.push(Value::Bool(a.is_truthy() || b.is_truthy()))?;
                }
                Instruction::Not => {
                    let a = self.stack.pop()?;
                    self.stack.push(a.not())?;
                }
                Instruction::Jump(target) => {
                    self.ip = target as usize;
                }
                Instruction::JumpIfTrue(target) => {
                    let condition = self.stack.peek()?;
                    if condition.is_truthy() {
                        self.ip = target as usize;
                    }
                }
                Instruction::JumpIfFalse(target) => {
                    let condition = self.stack.peek()?;
                    if !condition.is_truthy() {
                        self.ip = target as usize;
                    }
                }
                Instruction::JumpIfNone(target) => {
                    let value = self.stack.peek()?;
                    if matches!(value, Value::None) {
                        self.ip = target as usize;
                    }
                }
                Instruction::JumpBack(target) => {
                    self.ip = target as usize;
                }
                Instruction::Call(arg_count) => {
                    let func_idx = self.stack.len() - arg_count as usize - 1;
                    let func = self.stack.get(func_idx)?.clone();

                    match func {
                        Value::Function(f) => {
                            let frame = CallFrame {
                                chunk_index: self.chunk_index,
                                ip: self.ip,
                                stack_base: self.stack.len() - arg_count as usize,
                                upvalues: f.upvalues.clone(),
                                function_name: f.name.clone(),
                            };
                            self.frames.push(frame);
                            self.chunk_index = f.chunk_index;
                            self.ip = 0;
                        }
                        _ => {
                            return Err(OmegaError::TypeError {
                                message: format!("Cannot call {}", func.type_name()),
                                span: None,
                            });
                        }
                    }
                }
                Instruction::TailCall(arg_count) => {
                    // Tail call optimization: reuse current frame
                    let func_idx = self.stack.len() - arg_count as usize - 1;
                    let func = self.stack.get(func_idx)?.clone();

                    if let Value::Function(f) = func {
                        // Move arguments to base of stack
                        let base = self.frames.last().map(|f| f.stack_base).unwrap_or(0);
                        for i in 0..arg_count as usize {
                            let val = self.stack.pop()?;
                            self.stack.set(base + i, val)?;
                        }
                        self.stack.pop()?; // Remove function
                        self.chunk_index = f.chunk_index;
                        self.ip = 0;
                    }
                }
                Instruction::Return => {
                    let result = self.stack.pop()?;

                    // Execute defers
                    while let Some(&defer_ip) = self.defer_stack.last() {
                        self.defer_stack.pop();
                        // Execute defer block
                    }

                    if let Some(frame) = self.frames.pop() {
                        // Clean up stack
                        while self.stack.len() > frame.stack_base {
                            self.stack.pop()?;
                        }
                        self.stack.push(result)?;
                        self.chunk_index = frame.chunk_index;
                        self.ip = frame.ip;
                    } else {
                        return Ok(result);
                    }
                }
                Instruction::Yield => {
                    let value = self.stack.pop()?;
                    // Yield implementation for generators
                    self.stack.push(value)?;
                }
                Instruction::MakeClosure(constant_index, upvalue_indices) => {
                    let func = match &chunk.constants[constant_index as usize] {
                        Constant::Function(f) => f.clone(),
                        _ => return Err(OmegaError::InternalError {
                            message: "Expected function constant".to_string(),
                        }),
                    };

                    let mut upvalues = Vec::new();
                    for &idx in &upvalue_indices {
                        if let Some(frame) = self.frames.last() {
                            if let Some(upvalue) = frame.upvalues.get(idx as usize) {
                                upvalues.push(upvalue.clone());
                            }
                        }
                    }

                    let func_value = Value::Function(FunctionValue {
                        name: func.name,
                        chunk_index: func.chunk_index,
                        arity: func.arity,
                        upvalues,
                        is_async: func.is_async,
                    });
                    self.stack.push(func_value)?;
                }
                Instruction::MakeArray(count) => {
                    let mut elements = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        elements.push(self.stack.pop()?);
                    }
                    elements.reverse();
                    self.stack.push(Value::Array(elements))?;
                }
                Instruction::MakeMap(count) => {
                    let mut entries = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        let value = self.stack.pop()?;
                        let key = self.stack.pop()?;
                        entries.push((key, value));
                    }
                    entries.reverse();
                    self.stack.push(Value::Map(entries))?;
                }
                Instruction::MakeTuple(count) => {
                    let mut elements = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        elements.push(self.stack.pop()?);
                    }
                    elements.reverse();
                    self.stack.push(Value::Tuple(elements))?;
                }
                Instruction::MakeRange(inclusive) => {
                    let end = self.stack.pop()?;
                    let start = self.stack.pop()?;
                    match (start, end) {
                        (Value::Integer(s), Value::Integer(e)) => {
                            self.stack.push(Value::Range(s, e, inclusive))?;
                        }
                        _ => return Err(OmegaError::TypeError {
                            message: "Range requires integer bounds".to_string(),
                            span: None,
                        }),
                    }
                }
                Instruction::FormatString(arg_count) => {
                    let mut args = Vec::new();
                    for _ in 0..arg_count {
                        args.push(self.stack.pop()?);
                    }
                    args.reverse();
                    let format_str = match self.stack.pop()? {
                        Value::String(s) => s,
                        _ => return Err(OmegaError::TypeError {
                            message: "Expected format string".to_string(),
                            span: None,
                        }),
                    };
                    let result = self.format_string(&format_str, &args);
                    self.stack.push(Value::String(result))?;
                }
                Instruction::StringConcat => {
                    let b = self.stack.pop()?;
                    let a = self.stack.pop()?;
                    let result = format!("{}{}", a.format_display(), b.format_display());
                    self.stack.push(Value::String(result))?;
                }
                Instruction::GetIter => {
                    let iterable = self.stack.pop()?;
                    let iterator = iterable.iter()?;
                    self.stack.push(Value::Iterator(iterator))?;
                }
                Instruction::IterNext => {
                    if let Value::Iterator(ref mut iter) = self.stack.peek_mut()? {
                        if iter.index < iter.values.len() {
                            let value = iter.values[iter.index].clone();
                            iter.index += 1;
                            self.stack.push(value)?;
                        } else {
                            self.stack.push(Value::None)?;
                        }
                    }
                }
                Instruction::IterHasMore => {
                    if let Value::Iterator(iter) = self.stack.peek()? {
                        if iter.index < iter.values.len() {
                            self.stack.push(Value::Bool(true))?;
                        } else {
                            self.stack.push(Value::Bool(false))?;
                        }
                    }
                }
                Instruction::PushCatch(handler_ip) => {
                    self.catch_stack.push(CatchFrame {
                        handler_ip: handler_ip as usize,
                        chunk_index: self.chunk_index,
                    });
                }
                Instruction::PopCatch => {
                    self.catch_stack.pop();
                }
                Instruction::Throw => {
                    let error = self.stack.pop()?;
                    if let Some(catch_frame) = self.catch_stack.pop() {
                        self.ip = catch_frame.handler_ip;
                        self.chunk_index = catch_frame.chunk_index;
                        self.stack.push(error)?;
                    } else {
                        return Err(OmegaError::RuntimeError {
                            message: error.format_display(),
                            span: None,
                        });
                    }
                }
                Instruction::CastType(index) => {
                    let target_type = &chunk.constants[index as usize];
                    let value = self.stack.pop()?;
                    let result = self.cast_value(value, target_type)?;
                    self.stack.push(result)?;
                }
                Instruction::CheckType(index) => {
                    let check_type = &chunk.constants[index as usize];
                    let value = self.stack.peek()?;
                    let matches = self.check_type(value, check_type);
                    self.stack.push(Value::Bool(matches))?;
                }
                Instruction::TypeOf => {
                    let value = self.stack.pop()?;
                    self.stack.push(Value::String(value.type_name().to_string()))?;
                }
                Instruction::Print(newline) => {
                    let value = self.stack.pop()?;
                    if newline {
                        println!("{}", value.format_display());
                    } else {
                        print!("{}", value.format_display());
                    }
                    self.stack.push(Value::None)?;
                }
                Instruction::Assert => {
                    let message = self.stack.pop()?;
                    let condition = self.stack.pop()?;
                    if !condition.is_truthy() {
                        return Err(OmegaError::AssertionError {
                            message: message.format_display(),
                        });
                    }
                }
                Instruction::Nop => {}
                Instruction::Breakpoint => {
                    if self.debug {
                        eprintln!("Breakpoint at {}", self.ip - 1);
                    }
                }
                Instruction::Halt => {
                    return self.stack.pop().or_else(|_| Ok(Value::None));
                }
                Instruction::Defer(ip) => {
                    self.defer_stack.push(ip as usize);
                }
                Instruction::Drop => {
                    self.stack.pop()?;
                }
                Instruction::IncRef => {}
                Instruction::DecRef => {}
                _ => {
                    return Err(OmegaError::InternalError {
                        message: format!("Unimplemented instruction: {:?}", instruction),
                    });
                }
            }

            // Check for GC
            if self.heap.should_gc() {
                let roots = self.stack_roots();
                self.heap.gc(&roots);
            }
        }
    }

    fn constant_to_value(&self, constant: Constant, chunks: &[Bytecode]) -> OmegaResult<Value> {
        match constant {
            Constant::None => Ok(Value::None),
            Constant::Bool(v) => Ok(Value::Bool(v)),
            Constant::Integer(v) => Ok(Value::Integer(v)),
            Constant::Float(v) => Ok(Value::Float(v)),
            Constant::String(v) => Ok(Value::String(v)),
            Constant::Char(v) => Ok(Value::Char(v)),
            Constant::Byte(v) => Ok(Value::Byte(v)),
            Constant::BigInt(v) => Ok(Value::Integer(v.parse().unwrap_or(0))),
            Constant::Function(f) => Ok(Value::Function(FunctionValue {
                name: f.name,
                chunk_index: f.chunk_index,
                arity: f.arity,
                upvalues: Vec::new(),
                is_async: f.is_async,
            })),
            _ => Ok(Value::None),
        }
    }

    fn find_global(&self, name: &str, chunks: &[Bytecode]) -> OmegaResult<usize> {
        // Simple linear search for now
        for (i, chunk) in chunks.iter().enumerate() {
            for (j, constant) in chunk.constants.iter().enumerate() {
                if let Constant::String(s) = constant {
                    if s == name {
                        return Ok(j);
                    }
                }
            }
        }
        Err(OmegaError::NameError {
            name: name.to_string(),
            span: None,
        })
    }

    fn find_or_create_global(&mut self, name: &str) -> usize {
        // For now, just use the globals vector
        self.globals.len()
    }

    fn format_string(&self, format: &str, args: &[Value]) -> String {
        let mut result = String::new();
        let mut arg_index = 0;
        let mut chars = format.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                if chars.peek() == Some(&'}') {
                    chars.next();
                    if arg_index < args.len() {
                        result.push_str(&args[arg_index].format_display());
                        arg_index += 1;
                    }
                } else {
                    // Parse format spec
                    let mut spec = String::new();
                    while let Some(&c) = chars.peek() {
                        if c == '}' {
                            chars.next();
                            break;
                        }
                        spec.push(c);
                        chars.next();
                    }
                    if arg_index < args.len() {
                        result.push_str(&args[arg_index].format_display());
                        arg_index += 1;
                    }
                }
            } else if ch == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('t') => result.push('\t'),
                    Some('r') => result.push('\r'),
                    Some('\\') => result.push('\\'),
                    Some(c) => result.push(c),
                    None => {}
                }
            } else {
                result.push(ch);
            }
        }
        result
    }

    fn cast_value(&self, value: Value, target: &Constant) -> OmegaResult<Value> {
        match target {
            Constant::String(_) => Ok(Value::String(value.format_display())),
            Constant::Integer(_) => match value {
                Value::Float(f) => Ok(Value::Integer(f as i64)),
                Value::String(s) => Ok(Value::Integer(s.parse().map_err(|_| OmegaError::ValueError {
                    message: format!("Cannot convert '{}' to integer", s),
                })?)),
                Value::Bool(b) => Ok(Value::Integer(if b { 1 } else { 0 })),
                _ => Ok(value),
            },
            Constant::Float(_) => match value {
                Value::Integer(i) => Ok(Value::Float(i as f64)),
                Value::String(s) => Ok(Value::Float(s.parse().map_err(|_| OmegaError::ValueError {
                    message: format!("Cannot convert '{}' to float", s),
                })?)),
                _ => Ok(value),
            },
            Constant::Bool(_) => Ok(Value::Bool(value.is_truthy())),
            _ => Ok(value),
        }
    }

    fn check_type(&self, value: &Value, type_const: &Constant) -> bool {
        match type_const {
            Constant::String(type_name) => {
                match type_name.as_str() {
                    "i8" | "i16" | "i32" | "i64" | "i128" => matches!(value, Value::Integer(_)),
                    "f32" | "f64" => matches!(value, Value::Float(_)),
                    "bool" => matches!(value, Value::Bool(_)),
                    "char" => matches!(value, Value::Char(_)),
                    "String" => matches!(value, Value::String(_)),
                    "Array" => matches!(value, Value::Array(_)),
                    "Map" => matches!(value, Value::Map(_)),
                    "Tuple" => matches!(value, Value::Tuple(_)),
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn stack_roots(&self) -> Vec<Value> {
        self.stack.drain(self.stack.len())
    }

    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn heap_allocated(&self) -> usize {
        self.heap.allocated_count()
    }
}
