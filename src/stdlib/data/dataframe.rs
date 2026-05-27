use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<Value>),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{}", s),
            Value::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct DataFrame {
    columns: HashMap<String, Vec<Value>>,
    column_order: Vec<String>,
    length: usize,
}

impl DataFrame {
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
            column_order: Vec::new(),
            length: 0,
        }
    }

    pub fn with_capacity(rows: usize) -> Self {
        Self {
            columns: HashMap::new(),
            column_order: Vec::new(),
            length: 0,
        }
    }

    pub fn add_column(&mut self, name: &str, values: Vec<Value>) -> Result<(), String> {
        if !self.columns.is_empty() && values.len() != self.length {
            return Err(format!(
                "Column length {} doesn't match DataFrame length {}",
                values.len(),
                self.length
            ));
        }

        if self.columns.is_empty() {
            self.length = values.len();
        }

        self.column_order.push(name.to_string());
        self.columns.insert(name.to_string(), values);
        Ok(())
    }

    pub fn get_column(&self, name: &str) -> Option<&Vec<Value>> {
        self.columns.get(name)
    }

    pub fn get_row(&self, index: usize) -> Option<Vec<&Value>> {
        if index >= self.length {
            return None;
        }

        Some(
            self.column_order
                .iter()
                .map(|col| &self.columns[col][index])
                .collect(),
        )
    }

    pub fn get_value(&self, column: &str, row: usize) -> Option<&Value> {
        self.columns.get(column).and_then(|col| col.get(row))
    }

    pub fn set_value(&mut self, column: &str, row: usize, value: Value) -> Result<(), String> {
        if let Some(col) = self.columns.get_mut(column) {
            if row < col.len() {
                col[row] = value;
                Ok(())
            } else {
                Err(format!("Row index {} out of bounds", row))
            }
        } else {
            Err(format!("Column '{}' not found", column))
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn columns(&self) -> &[String] {
        &self.column_order
    }

    pub fn num_columns(&self) -> usize {
        self.column_order.len()
    }

    pub fn select(&self, columns: &[&str]) -> Result<DataFrame, String> {
        let mut result = DataFrame::new();

        for &col in columns {
            if let Some(values) = self.columns.get(col) {
                result.add_column(col, values.clone())?;
            } else {
                return Err(format!("Column '{}' not found", col));
            }
        }

        Ok(result)
    }

    pub fn filter<F>(&self, predicate: F) -> DataFrame
    where
        F: Fn(&DataFrame, usize) -> bool,
    {
        let mut result = DataFrame::new();
        let mut indices = Vec::new();

        for i in 0..self.length {
            if predicate(self, i) {
                indices.push(i);
            }
        }

        for col_name in &self.column_order {
            let col = &self.columns[col_name];
            let filtered: Vec<Value> = indices.iter().map(|&i| col[i].clone()).collect();
            result.add_column(col_name, filtered).unwrap();
        }

        result
    }

    pub fn sort_by(&self, column: &str, ascending: bool) -> Result<DataFrame, String> {
        if !self.columns.contains_key(column) {
            return Err(format!("Column '{}' not found", column));
        }

        let mut indices: Vec<usize> = (0..self.length).collect();
        let col = &self.columns[column];

        indices.sort_by(|&a, &b| {
            let cmp = compare_values(&col[a], &col[b]);
            if ascending {
                cmp
            } else {
                cmp.reverse()
            }
        });

        let mut result = DataFrame::new();
        for col_name in &self.column_order {
            let col = &self.columns[col_name];
            let sorted: Vec<Value> = indices.iter().map(|&i| col[i].clone()).collect();
            result.add_column(col_name, sorted).unwrap();
        }

        Ok(result)
    }

    pub fn group_by(&self, column: &str) -> Result<HashMap<String, DataFrame>, String> {
        if !self.columns.contains_key(column) {
            return Err(format!("Column '{}' not found", column));
        }

        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        let col = &self.columns[column];

        for i in 0..self.length {
            let key = format!("{}", col[i]);
            groups.entry(key).or_insert_with(Vec::new).push(i);
        }

        let mut result = HashMap::new();
        for (key, indices) in groups {
            let mut group_df = DataFrame::new();
            for col_name in &self.column_order {
                let col = &self.columns[col_name];
                let values: Vec<Value> = indices.iter().map(|&i| col[i].clone()).collect();
                group_df.add_column(col_name, values).unwrap();
            }
            result.insert(key, group_df);
        }

        Ok(result)
    }

    pub fn head(&self, n: usize) -> DataFrame {
        let limit = n.min(self.length);
        let mut result = DataFrame::new();

        for col_name in &self.column_order {
            let col = &self.columns[col_name];
            let values: Vec<Value> = col[..limit].to_vec();
            result.add_column(col_name, values).unwrap();
        }

        result
    }

    pub fn tail(&self, n: usize) -> DataFrame {
        let limit = n.min(self.length);
        let start = self.length - limit;
        let mut result = DataFrame::new();

        for col_name in &self.column_order {
            let col = &self.columns[col_name];
            let values: Vec<Value> = col[start..].to_vec();
            result.add_column(col_name, values).unwrap();
        }

        result
    }

    pub fn describe(&self) -> DataFrame {
        let mut result = DataFrame::new();
        let mut stats_names = Vec::new();
        let mut stats_values: HashMap<String, Vec<Value>> = HashMap::new();

        for col_name in &self.column_order {
            let col = &self.columns[col_name];

            let numeric_values: Vec<f64> = col
                .iter()
                .filter_map(|v| match v {
                    Value::Int(n) => Some(*n as f64),
                    Value::Float(n) => Some(*n),
                    _ => None,
                })
                .collect();

            if !numeric_values.is_empty() {
                let count = numeric_values.len() as f64;
                let sum: f64 = numeric_values.iter().sum();
                let mean = sum / count;
                let min = numeric_values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = numeric_values
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);

                let variance = numeric_values
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / count;
                let std = variance.sqrt();

                stats_values
                    .entry("count".to_string())
                    .or_insert_with(Vec::new)
                    .push(Value::Float(count));
                stats_values
                    .entry("mean".to_string())
                    .or_insert_with(Vec::new)
                    .push(Value::Float(mean));
                stats_values
                    .entry("std".to_string())
                    .or_insert_with(Vec::new)
                    .push(Value::Float(std));
                stats_values
                    .entry("min".to_string())
                    .or_insert_with(Vec::new)
                    .push(Value::Float(min));
                stats_values
                    .entry("max".to_string())
                    .or_insert_with(Vec::new)
                    .push(Value::Float(max));
            }
        }

        for (stat_name, values) in stats_values {
            result.add_column(&stat_name, values).unwrap();
        }

        result
    }

    pub fn merge(&self, other: &DataFrame) -> DataFrame {
        let mut result = DataFrame::new();

        for col_name in &self.column_order {
            if let Some(values) = self.columns.get(col_name) {
                result
                    .add_column(col_name, values.clone())
                    .unwrap();
            }
        }

        for col_name in &other.column_order {
            if !self.columns.contains_key(col_name) {
                if let Some(values) = other.columns.get(col_name) {
                    result
                        .add_column(col_name, values.clone())
                        .unwrap();
                }
            }
        }

        result
    }

    pub fn to_csv(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&self.column_order.join(","));
        output.push('\n');

        // Rows
        for i in 0..self.length {
            let row: Vec<String> = self
                .column_order
                .iter()
                .map(|col| {
                    let value = &self.columns[col][i];
                    match value {
                        Value::String(s) => format!("\"{}\"", s.replace('"', "\"\"")),
                        _ => format!("{}", value),
                    }
                })
                .collect();
            output.push_str(&row.join(","));
            output.push('\n');
        }

        output
    }

    pub fn from_csv(csv: &str) -> Result<DataFrame, String> {
        let mut lines = csv.lines();
        let header = lines
            .next()
            .ok_or("Empty CSV")?
            .split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>();

        let mut df = DataFrame::new();
        let mut columns: Vec<Vec<Value>> = header.iter().map(|_| Vec::new()).collect();

        for line in lines {
            let values: Vec<&str> = line.split(',').collect();
            if values.len() != header.len() {
                return Err(format!(
                    "Expected {} columns, got {}",
                    header.len(),
                    values.len()
                ));
            }

            for (i, value) in values.iter().enumerate() {
                let parsed = parse_value(value.trim());
                columns[i].push(parsed);
            }
        }

        for (i, col_name) in header.iter().enumerate() {
            df.add_column(col_name, std::mem::take(&mut columns[i]))?;
        }

        Ok(df)
    }

    pub fn to_json(&self) -> String {
        let mut rows = Vec::new();

        for i in 0..self.length {
            let mut row = Vec::new();
            for col_name in &self.column_order {
                let value = &self.columns[col_name][i];
                row.push(format!("\"{}\": {}", col_name, value));
            }
            rows.push(format!("{{ {} }}", row.join(", ")));
        }

        format!("[{}]", rows.join(", "))
    }
}

fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    match (a, b) {
        (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
        (Value::Null, _) => std::cmp::Ordering::Less,
        (_, Value::Null) => std::cmp::Ordering::Greater,
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        _ => std::cmp::Ordering::Equal,
    }
}

fn parse_value(s: &str) -> Value {
    if s.is_empty() || s == "null" || s == "none" {
        Value::Null
    } else if s == "true" {
        Value::Bool(true)
    } else if s == "false" {
        Value::Bool(false)
    } else if let Ok(n) = s.parse::<i64>() {
        Value::Int(n)
    } else if let Ok(n) = s.parse::<f64>() {
        Value::Float(n)
    } else {
        Value::String(s.to_string())
    }
}

impl std::fmt::Display for DataFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Calculate column widths
        let mut widths: Vec<usize> = self.column_order.iter().map(|c| c.len()).collect();

        for (i, col_name) in self.column_order.iter().enumerate() {
            let col = &self.columns[col_name];
            for value in col.iter().take(20) {
                let len = format!("{}", value).len();
                widths[i] = widths[i].max(len);
            }
        }

        // Header
        for (i, col_name) in self.column_order.iter().enumerate() {
            if i > 0 {
                write!(f, " | ")?;
            }
            write!(f, "{:width$}", col_name, width = widths[i])?;
        }
        writeln!(f)?;

        // Separator
        for (i, width) in widths.iter().enumerate() {
            if i > 0 {
                write!(f, "-+-")?;
            }
            write!(f, "{}", "-".repeat(*width))?;
        }
        writeln!(f)?;

        // Rows
        let display_rows = self.length.min(20);
        for i in 0..display_rows {
            for (j, col_name) in self.column_order.iter().enumerate() {
                if j > 0 {
                    write!(f, " | ")?;
                }
                let value = &self.columns[col_name][i];
                write!(f, "{:width$}", format!("{}", value), width = widths[j])?;
            }
            writeln!(f)?;
        }

        if self.length > 20 {
            writeln!(f, "... ({} more rows)", self.length - 20)?;
        }

        Ok(())
    }
}
