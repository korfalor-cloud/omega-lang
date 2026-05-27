use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;
use serde_json;

pub fn stringify(value: &Value) -> OmegaResult<String> {
    let json_value = omega_to_json(value);
    serde_json::to_string(&json_value).map_err(|e| OmegaError::FormatError {
        message: format!("JSON serialization error: {}", e),
    })
}

pub fn stringify_pretty(value: &Value) -> OmegaResult<String> {
    let json_value = omega_to_json(value);
    serde_json::to_string_pretty(&json_value).map_err(|e| OmegaError::FormatError {
        message: format!("JSON serialization error: {}", e),
    })
}

pub fn parse(s: &str) -> OmegaResult<Value> {
    let json_value: serde_json::Value = serde_json::from_str(s).map_err(|e| OmegaError::FormatError {
        message: format!("JSON parse error: {}", e),
    })?;
    json_to_omega(&json_value)
}

fn omega_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::None => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Integer(n) => serde_json::Value::Number(serde_json::Number::from(*n)),
        Value::Float(f) => {
            if let Some(n) = serde_json::Number::from_f64(*f) {
                serde_json::Value::Number(n)
            } else {
                serde_json::Value::Null
            }
        }
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Array(arr) => {
            let items: Vec<serde_json::Value> = arr.iter().map(omega_to_json).collect();
            serde_json::Value::Array(items)
        }
        Value::Map(map) => {
            let mut obj = serde_json::Map::new();
            for (key, val) in map {
                if let Value::String(k) = key {
                    obj.insert(k.clone(), omega_to_json(val));
                }
            }
            serde_json::Value::Object(obj)
        }
        Value::Tuple(tuple) => {
            let items: Vec<serde_json::Value> = tuple.iter().map(omega_to_json).collect();
            serde_json::Value::Array(items)
        }
        _ => serde_json::Value::String(value.format_display()),
    }
}

fn json_to_omega(value: &serde_json::Value) -> OmegaResult<Value> {
    match value {
        serde_json::Value::Null => Ok(Value::None),
        serde_json::Value::Bool(b) => Ok(Value::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::Float(f))
            } else {
                Ok(Value::None)
            }
        }
        serde_json::Value::String(s) => Ok(Value::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.iter().map(json_to_omega).collect::<OmegaResult<Vec<_>>>()?;
            Ok(Value::Array(items))
        }
        serde_json::Value::Object(obj) => {
            let mut map = Vec::new();
            for (key, val) in obj {
                map.push((Value::String(key.clone()), json_to_omega(val)?));
            }
            Ok(Value::Map(map))
        }
    }
}

pub fn get(json: &str, path: &str) -> OmegaResult<Value> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| OmegaError::FormatError {
        message: format!("JSON parse error: {}", e),
    })?;

    let mut current = &value;
    for part in path.split('.') {
        current = current.get(part).ok_or_else(|| OmegaError::KeyError {
            key: part.to_string(),
        })?;
    }
    json_to_omega(current)
}

pub fn set(json: &str, path: &str, value: &Value) -> OmegaResult<String> {
    let mut root: serde_json::Value = serde_json::from_str(json).map_err(|e| OmegaError::FormatError {
        message: format!("JSON parse error: {}", e),
    })?;

    let parts: Vec<&str> = path.split('.').collect();
    let mut current = &mut root;
    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            if let serde_json::Value::Object(obj) = current {
                obj.insert(part.to_string(), omega_to_json(value));
            }
        } else {
            current = current.get_mut(*part).ok_or_else(|| OmegaError::KeyError {
                key: part.to_string(),
            })?;
        }
    }

    serde_json::to_string(&root).map_err(|e| OmegaError::FormatError {
        message: format!("JSON serialization error: {}", e),
    })
}

pub fn merge(a: &str, b: &str) -> OmegaResult<String> {
    let mut va: serde_json::Value = serde_json::from_str(a).map_err(|e| OmegaError::FormatError {
        message: format!("JSON parse error: {}", e),
    })?;
    let vb: serde_json::Value = serde_json::from_str(b).map_err(|e| OmegaError::FormatError {
        message: format!("JSON parse error: {}", e),
    })?;

    merge_values(&mut va, &vb);

    serde_json::to_string(&va).map_err(|e| OmegaError::FormatError {
        message: format!("JSON serialization error: {}", e),
    })
}

fn merge_values(a: &mut serde_json::Value, b: &serde_json::Value) {
    match (a, b) {
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            for (key, value) in b {
                if let Some(a_val) = a.get_mut(key) {
                    merge_values(a_val, value);
                } else {
                    a.insert(key.clone(), value.clone());
                }
            }
        }
        (a, b) => {
            *a = b.clone();
        }
    }
}
