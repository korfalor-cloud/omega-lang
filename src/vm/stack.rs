use crate::errors::{OmegaError, OmegaResult};
use super::heap::Object;

#[derive(Debug, Clone)]
pub enum Value {
    None,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Char(char),
    Byte(u8),
    Array(Vec<Value>),
    Map(Vec<(Value, Value)>),
    Tuple(Vec<Value>),
    Object(Object),
    Function(FunctionValue),
    Iterator(IteratorValue),
    Range(i64, i64, bool),
}

#[derive(Debug, Clone)]
pub struct FunctionValue {
    pub name: String,
    pub chunk_index: usize,
    pub arity: u16,
    pub upvalues: Vec<Value>,
    pub is_async: bool,
}

#[derive(Debug, Clone)]
pub struct IteratorValue {
    pub values: Vec<Value>,
    pub index: usize,
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::None => false,
            Value::Bool(b) => *b,
            Value::Integer(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Map(m) => !m.is_empty(),
            Value::Tuple(t) => !t.is_empty(),
            _ => true,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::None => "None",
            Value::Bool(_) => "bool",
            Value::Integer(_) => "i64",
            Value::Float(_) => "f64",
            Value::String(_) => "String",
            Value::Char(_) => "char",
            Value::Byte(_) => "u8",
            Value::Array(_) => "Array",
            Value::Map(_) => "Map",
            Value::Tuple(_) => "Tuple",
            Value::Object(_) => "Object",
            Value::Function(_) => "Function",
            Value::Iterator(_) => "Iterator",
            Value::Range(_, _, _) => "Range",
        }
    }

    pub fn add(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a + b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a + *b as f64)),
            (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
            (Value::String(a), b) => Ok(Value::String(format!("{}{}", a, b))),
            (a, Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
            (Value::Array(a), Value::Array(b)) => {
                let mut result = a.clone();
                result.extend(b.iter().cloned());
                Ok(Value::Array(result))
            }
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot add {} and {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn sub(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a - *b as f64)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot subtract {} from {}", other.type_name(), self.type_name()),
                span: None,
            }),
        }
    }

    pub fn mul(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a * b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a * *b as f64)),
            (Value::String(s), Value::Integer(n)) => Ok(Value::String(s.repeat(*n as usize))),
            (Value::Array(a), Value::Integer(n)) => {
                let mut result = Vec::new();
                for _ in 0..*n {
                    result.extend(a.iter().cloned());
                }
                Ok(Value::Array(result))
            }
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot multiply {} and {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn div(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (_, Value::Integer(0)) | (_, Value::Float(0.0)) => Err(OmegaError::DivisionByZero),
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a / b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a / *b as f64)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot divide {} by {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn modulo(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (_, Value::Integer(0)) | (_, Value::Float(0.0)) => Err(OmegaError::DivisionByZero),
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a % b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot modulo {} by {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn pow(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a.pow(*b as u32))),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.powf(*b))),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float((*a as f64).powf(*b))),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a.powi(*b as i32))),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot raise {} to power {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn neg(&self) -> OmegaResult<Value> {
        match self {
            Value::Integer(v) => Ok(Value::Integer(-v)),
            Value::Float(v) => Ok(Value::Float(-v)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot negate {}", self.type_name()),
                span: None,
            }),
        }
    }

    pub fn eq(&self, other: &Value) -> Value {
        Value::Bool(self == other)
    }

    pub fn ne(&self, other: &Value) -> Value {
        Value::Bool(self != other)
    }

    pub fn lt(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Bool(a < b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Bool(a < b)),
            (Value::Char(a), Value::Char(b)) => Ok(Value::Bool(a < b)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot compare {} and {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn le(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Bool(a <= b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Bool(a <= b)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot compare {} and {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn gt(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Bool(a > b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Bool(a > b)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot compare {} and {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn ge(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Bool(a >= b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Bool(a >= b)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot compare {} and {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn bit_and(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a & b)),
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a & *b)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot bitwise AND {} and {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn bit_or(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a | b)),
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a | *b)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot bitwise OR {} and {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn bit_xor(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a ^ b)),
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a ^ *b)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot bitwise XOR {} and {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn bit_not(&self) -> OmegaResult<Value> {
        match self {
            Value::Integer(v) => Ok(Value::Integer(!v)),
            Value::Bool(v) => Ok(Value::Bool(!v)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot bitwise NOT {}", self.type_name()),
                span: None,
            }),
        }
    }

    pub fn shl(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a << b)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot shift left {} by {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn shr(&self, other: &Value) -> OmegaResult<Value> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a >> b)),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot shift right {} by {}", self.type_name(), other.type_name()),
                span: None,
            }),
        }
    }

    pub fn not(&self) -> Value {
        Value::Bool(!self.is_truthy())
    }

    pub fn format_display(&self) -> String {
        match self {
            Value::None => "none".to_string(),
            Value::Bool(v) => v.to_string(),
            Value::Integer(v) => v.to_string(),
            Value::Float(v) => v.to_string(),
            Value::String(v) => v.clone(),
            Value::Char(v) => v.to_string(),
            Value::Byte(v) => format!("{:#x}", v),
            Value::Array(elements) => {
                let items: Vec<String> = elements.iter().map(|v| v.format_display()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Map(entries) => {
                let items: Vec<String> = entries.iter()
                    .map(|(k, v)| format!("{}: {}", k.format_display(), v.format_display()))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
            Value::Tuple(elements) => {
                let items: Vec<String> = elements.iter().map(|v| v.format_display()).collect();
                format!("({})", items.join(", "))
            }
            Value::Object(obj) => format!("<object {}>", obj.type_name),
            Value::Function(func) => format!("<fn {}>", func.name),
            Value::Iterator(_) => "<iterator>".to_string(),
            Value::Range(start, end, inclusive) => {
                if *inclusive {
                    format!("{}..={}", start, end)
                } else {
                    format!("{}..{}", start, end)
                }
            }
        }
    }

    pub fn index(&self, index: &Value) -> OmegaResult<Value> {
        match (self, index) {
            (Value::Array(arr), Value::Integer(i)) => {
                let idx = if *i < 0 { (arr.len() as i64 + i) as usize } else { *i as usize };
                arr.get(idx).cloned().ok_or_else(|| OmegaError::IndexOutOfBounds {
                    index: *i,
                    length: arr.len(),
                })
            }
            (Value::String(s), Value::Integer(i)) => {
                let chars: Vec<char> = s.chars().collect();
                let idx = if *i < 0 { (chars.len() as i64 + i) as usize } else { *i as usize };
                chars.get(idx).map(|c| Value::Char(*c)).ok_or_else(|| OmegaError::IndexOutOfBounds {
                    index: *i,
                    length: chars.len(),
                })
            }
            (Value::Map(map), key) => {
                map.iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| OmegaError::KeyError {
                        key: key.format_display(),
                    })
            }
            (Value::Tuple(tuple), Value::Integer(i)) => {
                let idx = if *i < 0 { (tuple.len() as i64 + i) as usize } else { *i as usize };
                tuple.get(idx).cloned().ok_or_else(|| OmegaError::IndexOutOfBounds {
                    index: *i,
                    length: tuple.len(),
                })
            }
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot index {} with {}", self.type_name(), index.type_name()),
                span: None,
            }),
        }
    }

    pub fn slice(&self, start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> OmegaResult<Value> {
        match self {
            Value::Array(arr) => {
                let len = arr.len() as i64;
                let start = start.unwrap_or(0);
                let stop = stop.unwrap_or(len);
                let step = step.unwrap_or(1);

                let start = if start < 0 { len + start } else { start };
                let stop = if stop < 0 { len + stop } else { stop };

                if step > 0 {
                    let result: Vec<Value> = (start..stop)
                        .step_by(step as usize)
                        .filter_map(|i| arr.get(i as usize).cloned())
                        .collect();
                    Ok(Value::Array(result))
                } else {
                    let result: Vec<Value> = (stop..start)
                        .rev()
                        .step_by((-step) as usize)
                        .filter_map(|i| arr.get(i as usize).cloned())
                        .collect();
                    Ok(Value::Array(result))
                }
            }
            Value::String(s) => {
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len() as i64;
                let start = start.unwrap_or(0);
                let stop = stop.unwrap_or(len);
                let step = step.unwrap_or(1);

                let start = if start < 0 { len + start } else { start };
                let stop = if stop < 0 { len + stop } else { stop };

                let result: String = (start..stop)
                    .step_by(step as usize)
                    .filter_map(|i| chars.get(i as usize))
                    .collect();
                Ok(Value::String(result))
            }
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot slice {}", self.type_name()),
                span: None,
            }),
        }
    }

    pub fn iter(&self) -> OmegaResult<IteratorValue> {
        match self {
            Value::Array(arr) => Ok(IteratorValue {
                values: arr.clone(),
                index: 0,
            }),
            Value::String(s) => Ok(IteratorValue {
                values: s.chars().map(Value::Char).collect(),
                index: 0,
            }),
            Value::Range(start, end, inclusive) => {
                let end_val = if *inclusive { end + 1 } else { *end };
                Ok(IteratorValue {
                    values: (*start..end_val).map(Value::Integer).collect(),
                    index: 0,
                })
            }
            Value::Tuple(tuple) => Ok(IteratorValue {
                values: tuple.clone(),
                index: 0,
            }),
            _ => Err(OmegaError::TypeError {
                message: format!("Cannot iterate over {}", self.type_name()),
                span: None,
            }),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::None, Value::None) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::Map(a), Value::Map(b)) => a == b,
            _ => false,
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_display())
    }
}

pub struct Stack {
    values: Vec<Value>,
    max_size: usize,
}

impl Stack {
    pub fn new() -> Self {
        Self {
            values: Vec::with_capacity(256),
            max_size: 8192,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            max_size: 8192,
        }
    }

    pub fn push(&mut self, value: Value) -> OmegaResult<()> {
        if self.values.len() >= self.max_size {
            return Err(OmegaError::StackOverflow);
        }
        self.values.push(value);
        Ok(())
    }

    pub fn pop(&mut self) -> OmegaResult<Value> {
        self.values.pop().ok_or_else(|| OmegaError::RuntimeError {
            message: "Stack underflow".to_string(),
            span: None,
        })
    }

    pub fn peek(&self) -> OmegaResult<&Value> {
        self.values.last().ok_or_else(|| OmegaError::RuntimeError {
            message: "Stack is empty".to_string(),
            span: None,
        })
    }

    pub fn peek_mut(&mut self) -> OmegaResult<&mut Value> {
        self.values.last_mut().ok_or_else(|| OmegaError::RuntimeError {
            message: "Stack is empty".to_string(),
            span: None,
        })
    }

    pub fn dup(&mut self) -> OmegaResult<()> {
        let value = self.peek()?.clone();
        self.push(value)
    }

    pub fn swap(&mut self) -> OmegaResult<()> {
        let len = self.values.len();
        if len < 2 {
            return Err(OmegaError::RuntimeError {
                message: "Stack needs at least 2 values for swap".to_string(),
                span: None,
            });
        }
        self.values.swap(len - 1, len - 2);
        Ok(())
    }

    pub fn rot3(&mut self) -> OmegaResult<()> {
        let len = self.values.len();
        if len < 3 {
            return Err(OmegaError::RuntimeError {
                message: "Stack needs at least 3 values for rot3".to_string(),
                span: None,
            });
        }
        let c = self.values.pop().unwrap();
        let b = self.values.pop().unwrap();
        let a = self.values.pop().unwrap();
        self.values.push(b);
        self.values.push(c);
        self.values.push(a);
        Ok(())
    }

    pub fn get(&self, index: usize) -> OmegaResult<&Value> {
        self.values.get(index).ok_or_else(|| OmegaError::RuntimeError {
            message: format!("Stack index {} out of bounds", index),
            span: None,
        })
    }

    pub fn set(&mut self, index: usize, value: Value) -> OmegaResult<()> {
        if index >= self.values.len() {
            return Err(OmegaError::RuntimeError {
                message: format!("Stack index {} out of bounds", index),
                span: None,
            });
        }
        self.values[index] = value;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    pub fn drain(&mut self, count: usize) -> Vec<Value> {
        let start = self.values.len().saturating_sub(count);
        self.values.drain(start..).collect()
    }
}
