use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;

pub fn stringify(value: &Value) -> OmegaResult<String> {
    // Simplified YAML serialization
    Ok(value_to_yaml(value, 0))
}

fn value_to_yaml(value: &Value, indent: usize) -> String {
    let prefix = "  ".repeat(indent);
    match value {
        Value::None => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => {
            if s.contains('\n') {
                let lines: Vec<String> = s.lines().map(|l| format!("{}  {}", prefix, l)).collect();
                format!("|\n{}", lines.join("\n"))
            } else if s.contains(':') || s.contains('#') || s.contains('"') {
                format!("\"{}\"", s.replace('"', "\\\""))
            } else {
                s.clone()
            }
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                return "[]".to_string();
            }
            let items: Vec<String> = arr.iter()
                .map(|v| format!("\n{}- {}", prefix, value_to_yaml(v, indent + 1)))
                .collect();
            items.join("")
        }
        Value::Map(map) => {
            if map.is_empty() {
                return "{}".to_string();
            }
            let entries: Vec<String> = map.iter()
                .map(|(k, v)| {
                    let key = match k {
                        Value::String(s) => s.clone(),
                        _ => k.format_display(),
                    };
                    format!("\n{}{}: {}", prefix, key, value_to_yaml(v, indent + 1))
                })
                .collect();
            entries.join("")
        }
        Value::Tuple(tuple) => {
            let items: Vec<String> = tuple.iter()
                .map(|v| value_to_yaml(v, indent))
                .collect();
            format!("[{}]", items.join(", "))
        }
        _ => value.format_display(),
    }
}

pub fn parse(s: &str) -> OmegaResult<Value> {
    // Simplified YAML parser
    parse_yaml_value(s.trim())
}

fn parse_yaml_value(s: &str) -> OmegaResult<Value> {
    if s.is_empty() || s == "null" || s == "~" {
        return Ok(Value::None);
    }
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
        if inner.trim().is_empty() {
            return Ok(Value::Array(Vec::new()));
        }
        let items: Vec<Value> = inner.split(',')
            .map(|item| parse_yaml_value(item.trim()))
            .collect::<OmegaResult<Vec<_>>>()?;
        return Ok(Value::Array(items));
    }
    if s.contains(':') {
        let mut map = Vec::new();
        for line in s.lines() {
            if let Some(pos) = line.find(':') {
                let key = line[..pos].trim().to_string();
                let value = line[pos+1..].trim();
                map.push((Value::String(key), parse_yaml_value(value)?));
            }
        }
        return Ok(Value::Map(map));
    }
    Ok(Value::String(s.to_string()))
}
