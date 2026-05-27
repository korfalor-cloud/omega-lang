use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;

pub fn stringify(value: &Value) -> OmegaResult<String> {
    match value {
        Value::Map(map) => {
            let mut result = String::new();
            for (key, val) in map {
                if let Value::String(k) = key {
                    result.push_str(&format!("{} = {}\n", k, value_to_toml(val)));
                }
            }
            Ok(result)
        }
        _ => Err(OmegaError::FormatError {
            message: "TOML root must be a table".to_string(),
        }),
    }
}

fn value_to_toml(value: &Value) -> String {
    match value {
        Value::None => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(value_to_toml).collect();
            format!("[{}]", items.join(", "))
        }
        _ => "\"\"".to_string(),
    }
}

pub fn parse(s: &str) -> OmegaResult<Value> {
    let mut map = Vec::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
            continue;
        }
        if let Some(pos) = line.find('=') {
            let key = line[..pos].trim().to_string();
            let value = line[pos+1..].trim();
            map.push((Value::String(key), parse_toml_value(value)?));
        }
    }
    Ok(Value::Map(map))
}

fn parse_toml_value(s: &str) -> OmegaResult<Value> {
    if s == "true" {
        return Ok(Value::Bool(true));
    }
    if s == "false" {
        return Ok(Value::Bool(false));
    }
    if let Ok(n) = s.parse::<i64>() {
        return Ok(Value::Integer(n));
    }
    if let Ok(f) = s.parse::<f64>() {
        return Ok(Value::Float(f));
    }
    if s.starts_with('"') && s.ends_with('"') {
        return Ok(Value::String(s[1..s.len()-1].to_string()));
    }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len()-1];
        let items: Vec<Value> = inner.split(',')
            .map(|item| parse_toml_value(item.trim()))
            .collect::<OmegaResult<Vec<_>>>()?;
        return Ok(Value::Array(items));
    }
    Ok(Value::String(s.to_string()))
}
