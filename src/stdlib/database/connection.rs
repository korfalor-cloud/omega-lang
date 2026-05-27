/// Database connection abstraction.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Connection {
    pub url: String,
    pub pool_size: usize,
    pub timeout_ms: u64,
    pub options: HashMap<String, String>,
    connected: bool,
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub rows: Vec<HashMap<String, Value>>,
    pub affected_rows: usize,
    pub last_insert_id: Option<i64>,
}

#[derive(Debug, Clone)]
pub enum Value {
    Null,
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
    Bytes(Vec<u8>),
}

impl Value {
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Value::Integer(v) => Some(*v),
            Value::Boolean(v) => Some(if *v { 1 } else { 0 }),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(*v),
            Value::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Boolean(v) => Some(*v),
            Value::Integer(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Value::Null => "NULL".to_string(),
            Value::Integer(v) => v.to_string(),
            Value::Float(v) => v.to_string(),
            Value::Text(v) => v.clone(),
            Value::Boolean(v) => if *v { "true" } else { "false" }.to_string(),
            Value::Bytes(v) => format!("{:?}", v),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl Connection {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            pool_size: 10,
            timeout_ms: 30000,
            options: HashMap::new(),
            connected: false,
        }
    }

    pub fn pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    pub fn timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    pub fn connect(&mut self) -> Result<(), String> {
        // Simulated connection
        self.connected = true;
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.connected = false;
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn execute(&self, sql: &str, _params: &[Value]) -> Result<QueryResult, String> {
        if !self.connected {
            return Err("Not connected".to_string());
        }
        // Simulated execution
        Ok(QueryResult {
            rows: Vec::new(),
            affected_rows: 0,
            last_insert_id: None,
        })
    }

    pub fn query(&self, sql: &str, _params: &[Value]) -> Result<QueryResult, String> {
        if !self.connected {
            return Err("Not connected".to_string());
        }
        Ok(QueryResult {
            rows: Vec::new(),
            affected_rows: 0,
            last_insert_id: None,
        })
    }

    pub fn begin_transaction(&self) -> Result<Transaction, String> {
        if !self.connected {
            return Err("Not connected".to_string());
        }
        Ok(Transaction { active: true })
    }
}

#[derive(Debug)]
pub struct Transaction {
    active: bool,
}

impl Transaction {
    pub fn commit(&mut self) -> Result<(), String> {
        self.active = false;
        Ok(())
    }

    pub fn rollback(&mut self) -> Result<(), String> {
        self.active = false;
        Ok(())
    }

    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// Connection pool
pub struct ConnectionPool {
    connections: Vec<Connection>,
    max_size: usize,
}

impl ConnectionPool {
    pub fn new(url: &str, max_size: usize) -> Self {
        let connections = (0..max_size)
            .map(|_| Connection::new(url))
            .collect();
        Self { connections, max_size }
    }

    pub fn get(&mut self) -> Option<&mut Connection> {
        self.connections.iter_mut().find(|c| c.is_connected())
            .or_else(|| self.connections.first_mut())
    }

    pub fn size(&self) -> usize {
        self.connections.len()
    }

    pub fn close_all(&mut self) {
        for conn in &mut self.connections {
            conn.disconnect();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection() {
        let mut conn = Connection::new("sqlite:test.db");
        assert!(!conn.is_connected());
        conn.connect().unwrap();
        assert!(conn.is_connected());
    }

    #[test]
    fn test_value_types() {
        let v = Value::Integer(42);
        assert_eq!(v.as_integer(), Some(42));
        assert_eq!(v.as_float(), Some(42.0));

        let v = Value::Text("hello".to_string());
        assert_eq!(v.as_text(), Some("hello"));
    }

    #[test]
    fn test_transaction() {
        let conn = Connection::new("sqlite:test.db");
        let mut tx = conn.begin_transaction();
        assert!(tx.is_ok());
        tx.unwrap().commit().unwrap();
    }

    #[test]
    fn test_connection_pool() {
        let mut pool = ConnectionPool::new("sqlite:test.db", 5);
        assert_eq!(pool.size(), 5);
    }
}
