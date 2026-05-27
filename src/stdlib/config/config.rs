/// Configuration management system.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum ConfigValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<ConfigValue>),
    Object(HashMap<String, ConfigValue>),
}

impl ConfigValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            ConfigValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match self {
            ConfigValue::Integer(v) => Some(*v),
            ConfigValue::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConfigValue::Float(v) => Some(*v),
            ConfigValue::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Boolean(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<ConfigValue>> {
        match self {
            ConfigValue::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&HashMap<String, ConfigValue>> {
        match self {
            ConfigValue::Object(v) => Some(v),
            _ => None,
        }
    }

    pub fn to_string_value(&self) -> String {
        match self {
            ConfigValue::String(s) => s.clone(),
            ConfigValue::Integer(v) => v.to_string(),
            ConfigValue::Float(v) => v.to_string(),
            ConfigValue::Boolean(v) => v.to_string(),
            ConfigValue::Array(_) => "[array]".to_string(),
            ConfigValue::Object(_) => "[object]".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    values: HashMap<String, ConfigValue>,
    defaults: HashMap<String, ConfigValue>,
}

impl Config {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            defaults: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: ConfigValue) {
        self.values.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.values.get(key).or_else(|| self.defaults.get(key))
    }

    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_string()
    }

    pub fn get_integer(&self, key: &str) -> Option<i64> {
        self.get(key)?.as_integer()
    }

    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.get(key)?.as_float()
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    pub fn set_default(&mut self, key: &str, value: ConfigValue) {
        self.defaults.insert(key.to_string(), value);
    }

    pub fn has(&self, key: &str) -> bool {
        self.values.contains_key(key) || self.defaults.contains_key(key)
    }

    pub fn keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.values.keys().map(|s| s.as_str()).collect();
        for key in self.defaults.keys() {
            if !keys.contains(&key.as_str()) {
                keys.push(key.as_str());
            }
        }
        keys
    }

    pub fn merge(&mut self, other: &Config) {
        for (key, value) in &other.values {
            self.values.insert(key.clone(), value.clone());
        }
    }

    pub fn from_map(map: HashMap<String, String>) -> Self {
        let mut config = Config::new();
        for (key, value) in map {
            config.set(&key, ConfigValue::String(value));
        }
        config
    }

    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for key in self.keys() {
            if let Some(value) = self.get(key) {
                map.insert(key.to_string(), value.to_string_value());
            }
        }
        map
    }

    pub fn load_from_json(&mut self, json: &str) -> Result<(), String> {
        // Simple JSON parser for config
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err("Invalid JSON".to_string());
        }

        let inner = &json[1..json.len() - 1];
        for pair in inner.split(',') {
            let parts: Vec<&str> = pair.splitn(2, ':').collect();
            if parts.len() == 2 {
                let key = parts[0].trim().trim_matches('"');
                let value = parts[1].trim();
                if value.starts_with('"') && value.ends_with('"') {
                    self.set(key, ConfigValue::String(value[1..value.len() - 1].to_string()));
                } else if value == "true" {
                    self.set(key, ConfigValue::Boolean(true));
                } else if value == "false" {
                    self.set(key, ConfigValue::Boolean(false));
                } else if let Ok(n) = value.parse::<i64>() {
                    self.set(key, ConfigValue::Integer(n));
                } else if let Ok(f) = value.parse::<f64>() {
                    self.set(key, ConfigValue::Float(f));
                }
            }
        }

        Ok(())
    }

    pub fn load_from_toml(&mut self, toml: &str) -> Result<(), String> {
        let mut current_section = String::new();

        for line in toml.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with('[') && line.ends_with(']') {
                current_section = line[1..line.len() - 1].to_string();
                continue;
            }

            if let Some(eq_pos) = line.find('=') {
                let key = line[..eq_pos].trim();
                let value = line[eq_pos + 1..].trim().trim_matches('"');

                let full_key = if current_section.is_empty() {
                    key.to_string()
                } else {
                    format!("{}.{}", current_section, key)
                };

                if value == "true" {
                    self.set(&full_key, ConfigValue::Boolean(true));
                } else if value == "false" {
                    self.set(&full_key, ConfigValue::Boolean(false));
                } else if let Ok(n) = value.parse::<i64>() {
                    self.set(&full_key, ConfigValue::Integer(n));
                } else if let Ok(f) = value.parse::<f64>() {
                    self.set(&full_key, ConfigValue::Float(f));
                } else {
                    self.set(&full_key, ConfigValue::String(value.to_string()));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_basic() {
        let mut config = Config::new();
        config.set("name", ConfigValue::String("test".to_string()));
        config.set("port", ConfigValue::Integer(8080));

        assert_eq!(config.get_string("name"), Some("test"));
        assert_eq!(config.get_integer("port"), Some(8080));
    }

    #[test]
    fn test_config_defaults() {
        let mut config = Config::new();
        config.set_default("timeout", ConfigValue::Integer(30));

        assert_eq!(config.get_integer("timeout"), Some(30));

        config.set("timeout", ConfigValue::Integer(60));
        assert_eq!(config.get_integer("timeout"), Some(60));
    }

    #[test]
    fn test_config_merge() {
        let mut config1 = Config::new();
        config1.set("a", ConfigValue::Integer(1));

        let mut config2 = Config::new();
        config2.set("b", ConfigValue::Integer(2));

        config1.merge(&config2);
        assert_eq!(config1.get_integer("a"), Some(1));
        assert_eq!(config1.get_integer("b"), Some(2));
    }

    #[test]
    fn test_config_json() {
        let mut config = Config::new();
        config.load_from_json(r#"{"name": "test", "port": 8080}"#).unwrap();

        assert_eq!(config.get_string("name"), Some("test"));
        assert_eq!(config.get_integer("port"), Some(8080));
    }

    #[test]
    fn test_config_toml() {
        let mut config = Config::new();
        config.load_from_toml("[server]\nport = 8080\nhost = \"localhost\"").unwrap();

        assert_eq!(config.get_integer("server.port"), Some(8080));
        assert_eq!(config.get_string("server.host"), Some("localhost"));
    }
}
