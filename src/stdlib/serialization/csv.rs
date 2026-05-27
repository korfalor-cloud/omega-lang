use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;

pub fn parse(s: &str) -> OmegaResult<Vec<Vec<String>>> {
    let mut records = Vec::new();
    for line in s.lines() {
        let record = parse_csv_line(line);
        records.push(record);
    }
    Ok(records)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' => {
                if in_quotes {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        current.push('"');
                    } else {
                        in_quotes = false;
                    }
                } else {
                    in_quotes = true;
                }
            }
            ',' if !in_quotes => {
                fields.push(current);
                current = String::new();
            }
            _ => current.push(c),
        }
    }
    fields.push(current);
    fields
}

pub fn stringify(records: &[Vec<String>]) -> String {
    records.iter()
        .map(|record| {
            record.iter()
                .map(|field| {
                    if field.contains(',') || field.contains('"') || field.contains('\n') {
                        format!("\"{}\"", field.replace('"', "\"\""))
                    } else {
                        field.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn parse_to_values(s: &str) -> OmegaResult<Vec<Vec<Value>>> {
    let records = parse(s)?;
    Ok(records.into_iter()
        .map(|record| record.into_iter().map(Value::String).collect())
        .collect())
}

pub fn parse_with_headers(s: &str) -> OmegaResult<Vec<Value>> {
    let records = parse(s)?;
    if records.is_empty() {
        return Ok(Vec::new());
    }
    let headers = &records[0];
    let mut result = Vec::new();
    for record in &records[1..] {
        let mut map = Vec::new();
        for (i, field) in record.iter().enumerate() {
            if let Some(header) = headers.get(i) {
                map.push((Value::String(header.clone()), Value::String(field.clone())));
            }
        }
        result.push(Value::Map(map));
    }
    Ok(result)
}
