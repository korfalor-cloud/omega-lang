/// Advanced configuration parsing, validation, and merging.

use std::collections::HashMap;

use super::config::{Config, ConfigValue};

// ---------------------------------------------------------------------------
// TOML parser (extended – supports inline tables and arrays)
// ---------------------------------------------------------------------------

pub struct TomlParser;

impl TomlParser {
    pub fn parse(input: &str) -> Result<Config, String> {
        let mut config = Config::new();
        let mut section = String::new();

        for (lineno, raw) in input.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Section header  [a.b.c]
            if line.starts_with('[') && line.ends_with(']') {
                section = line[1..line.len() - 1].trim().to_string();
                continue;
            }

            let eq = line.find('=').ok_or_else(|| {
                format!("line {}: expected '=' in assignment", lineno + 1)
            })?;

            let key = line[..eq].trim();
            let raw_val = line[eq + 1..].trim();

            let full_key = if section.is_empty() {
                key.to_string()
            } else {
                format!("{}.{}", section, key)
            };

            let value = Self::parse_value(raw_val)?;
            config.set(&full_key, value);
        }

        Ok(config)
    }

    fn parse_value(raw: &str) -> Result<ConfigValue, String> {
        // Inline array  [1, 2, 3]
        if raw.starts_with('[') && raw.ends_with(']') {
            let inner = &raw[1..raw.len() - 1];
            let items: Vec<ConfigValue> = inner
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| Self::parse_value(s.trim()))
                .collect::<Result<_, _>>()?;
            return Ok(ConfigValue::Array(items));
        }

        // Inline table  { k = "v", n = 1 }
        if raw.starts_with('{') && raw.ends_with('}') {
            let inner = &raw[1..raw.len() - 1];
            let mut map = HashMap::new();
            for pair in inner.split(',') {
                let pair = pair.trim();
                if pair.is_empty() {
                    continue;
                }
                let eq = pair
                    .find('=')
                    .ok_or("expected '=' inside inline table")?;
                let k = pair[..eq].trim().trim_matches('"').to_string();
                let v = Self::parse_value(pair[eq + 1..].trim())?;
                map.insert(k, v);
            }
            return Ok(ConfigValue::Object(map));
        }

        // String
        if raw.starts_with('"') && raw.ends_with('"') {
            return Ok(ConfigValue::String(raw[1..raw.len() - 1].to_string()));
        }

        // Boolean
        if raw == "true" {
            return Ok(ConfigValue::Boolean(true));
        }
        if raw == "false" {
            return Ok(ConfigValue::Boolean(false));
        }

        // Integer then float
        if let Ok(n) = raw.parse::<i64>() {
            return Ok(ConfigValue::Integer(n));
        }
        if let Ok(f) = raw.parse::<f64>() {
            return Ok(ConfigValue::Float(f));
        }

        // Bare string fallback
        Ok(ConfigValue::String(raw.to_string()))
    }
}

// ---------------------------------------------------------------------------
// YAML parser (minimal subset: key-value, nesting, lists)
// ---------------------------------------------------------------------------

pub struct YamlParser;

impl YamlParser {
    pub fn parse(input: &str) -> Result<Config, String> {
        let mut config = Config::new();
        let lines: Vec<&str> = input.lines().collect();
        Self::parse_block(&lines, 0, &mut config, "")?;
        Ok(config)
    }

    fn parse_block(
        lines: &[&str],
        start: usize,
        config: &mut Config,
        prefix: &str,
    ) -> Result<usize, String> {
        let mut i = start;
        while i < lines.len() {
            let raw = lines[i];
            let indent = raw.chars().take_while(|c| *c == ' ').count();
            let trimmed = raw.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') {
                i += 1;
                continue;
            }

            // Determine the reference indent from the first meaningful line
            if i == start && trimmed.contains(':') {
                // ok
            }

            if let Some(colon) = trimmed.find(':') {
                let key = trimmed[..colon].trim();
                let val_part = trimmed[colon + 1..].trim();
                let full_key = if prefix.is_empty() {
                    key.to_string()
                } else {
                    format!("{}.{}", prefix, key)
                };

                if val_part.is_empty() {
                    // Nested map or list block
                    i += 1;
                    i = Self::parse_block(lines, i, config, &full_key)?;
                } else {
                    let value = Self::parse_scalar(val_part);
                    config.set(&full_key, value);
                    i += 1;
                }
            } else if trimmed.starts_with('-') {
                // List items collected under parent key
                let item_val = Self::parse_scalar(trimmed[1..].trim());
                // Append as indexed keys so we can round-trip
                let mut idx = 0usize;
                while config.has(&format!("{}[{}]", prefix, idx)) {
                    idx += 1;
                }
                config.set(&format!("{}[{}]", prefix, idx), item_val);
                i += 1;
            } else {
                i += 1;
            }

            // If indentation decreased we are done with this block
            if i < lines.len() {
                let next_raw = lines[i];
                let next_indent = next_raw.chars().take_while(|c| *c == ' ').count();
                if next_raw.trim().is_empty() {
                    // skip blanks
                } else if next_indent < indent && !next_raw.trim().starts_with('#') {
                    break;
                }
            }
        }
        Ok(i)
    }

    fn parse_scalar(raw: &str) -> ConfigValue {
        if (raw.starts_with('"') && raw.ends_with('"'))
            || (raw.starts_with('\'') && raw.ends_with('\''))
        {
            return ConfigValue::String(raw[1..raw.len() - 1].to_string());
        }
        if raw == "true" {
            return ConfigValue::Boolean(true);
        }
        if raw == "false" {
            return ConfigValue::Boolean(false);
        }
        if raw == "null" || raw == "~" {
            return ConfigValue::String(String::new());
        }
        if let Ok(n) = raw.parse::<i64>() {
            return ConfigValue::Integer(n);
        }
        if let Ok(f) = raw.parse::<f64>() {
            return ConfigValue::Float(f);
        }
        ConfigValue::String(raw.to_string())
    }
}

// ---------------------------------------------------------------------------
// Environment variable loader
// ---------------------------------------------------------------------------

pub struct EnvLoader {
    prefix: String,
    separator: String,
}

impl EnvLoader {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            separator: "_".to_string(),
        }
    }

    pub fn with_separator(mut self, sep: &str) -> Self {
        self.separator = sep.to_string();
        self
    }

    /// Build a Config from the supplied key-value pairs (simulated env).
    pub fn load_from_pairs(&self, pairs: &[(&str, &str)]) -> Config {
        let mut config = Config::new();
        let pfx = format!("{}{}", self.prefix, self.separator);
        for (k, v) in pairs {
            if let Some(suffix) = k.strip_prefix(&pfx) {
                let cfg_key = suffix
                    .to_lowercase()
                    .replace(&self.separator, ".");
                if let Ok(n) = v.parse::<i64>() {
                    config.set(&cfg_key, ConfigValue::Integer(n));
                } else if let Ok(f) = v.parse::<f64>() {
                    config.set(&cfg_key, ConfigValue::Float(f));
                } else if *v == "true" || *v == "false" {
                    config.set(&cfg_key, ConfigValue::Boolean(*v == "true"));
                } else {
                    config.set(&cfg_key, ConfigValue::String(v.to_string()));
                }
            }
        }
        config
    }
}

// ---------------------------------------------------------------------------
// Configuration validation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub key: String,
    pub required: bool,
    pub kind: ValueKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueKind {
    String,
    Integer,
    Float,
    Boolean,
    Any,
}

pub struct ConfigValidator {
    rules: Vec<ValidationRule>,
}

impl ConfigValidator {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn require(mut self, key: &str, kind: ValueKind) -> Self {
        self.rules.push(ValidationRule {
            key: key.to_string(),
            required: true,
            kind,
        });
        self
    }

    pub fn optional(mut self, key: &str, kind: ValueKind) -> Self {
        self.rules.push(ValidationRule {
            key: key.to_string(),
            required: false,
            kind,
        });
        self
    }

    pub fn validate(&self, config: &Config) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        for rule in &self.rules {
            match config.get(&rule.key) {
                None => {
                    if rule.required {
                        errors.push(format!("missing required key '{}'", rule.key));
                    }
                }
                Some(val) => {
                    let ok = match (&rule.kind, val) {
                        (ValueKind::Any, _) => true,
                        (ValueKind::String, ConfigValue::String(_)) => true,
                        (ValueKind::Integer, ConfigValue::Integer(_)) => true,
                        (ValueKind::Float, ConfigValue::Float(_) | ConfigValue::Integer(_)) => {
                            true
                        }
                        (ValueKind::Boolean, ConfigValue::Boolean(_)) => true,
                        _ => false,
                    };
                    if !ok {
                        errors.push(format!(
                            "key '{}' has wrong type (expected {:?})",
                            rule.key, rule.kind
                        ));
                    }
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// Nested config merging
// ---------------------------------------------------------------------------

/// Deep-merge `overlay` into `base`.  Overlay values win on conflict; nested
/// objects are merged recursively rather than replaced.
pub fn deep_merge(base: &mut Config, overlay: &Config) {
    for key in overlay.keys() {
        let key_owned = key.to_string();
        match (base.get(&key_owned), overlay.get(&key_owned)) {
            (
                Some(ConfigValue::Object(base_map)),
                Some(ConfigValue::Object(overlay_map)),
            ) => {
                let mut merged = base_map.clone();
                for (k, v) in overlay_map {
                    merged.insert(k.clone(), v.clone());
                }
                base.set(&key_owned, ConfigValue::Object(merged));
            }
            (_, Some(val)) => {
                base.set(&key_owned, val.clone());
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TOML parser --------------------------------------------------------

    #[test]
    fn test_toml_basic() {
        let toml = r#"
# comment
name = "omega"
port = 8080
debug = true
ratio = 3.14
"#;
        let cfg = TomlParser::parse(toml).unwrap();
        assert_eq!(cfg.get_string("name"), Some("omega"));
        assert_eq!(cfg.get_integer("port"), Some(8080));
        assert_eq!(cfg.get_bool("debug"), Some(true));
        assert!((cfg.get_float("ratio").unwrap() - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn test_toml_sections() {
        let toml = "[server]\nhost = \"0.0.0.0\"\nport = 443\n";
        let cfg = TomlParser::parse(toml).unwrap();
        assert_eq!(cfg.get_string("server.host"), Some("0.0.0.0"));
        assert_eq!(cfg.get_integer("server.port"), Some(443));
    }

    #[test]
    fn test_toml_inline_array() {
        let toml = "ports = [80, 443, 8080]\n";
        let cfg = TomlParser::parse(toml).unwrap();
        let arr = cfg.get("ports").unwrap().as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0].as_integer(), Some(80));
    }

    #[test]
    fn test_toml_inline_table() {
        let toml = "point = { x = 1, y = 2 }\n";
        let cfg = TomlParser::parse(toml).unwrap();
        let obj = cfg.get("point").unwrap().as_object().unwrap();
        assert_eq!(obj.get("x").unwrap().as_integer(), Some(1));
        assert_eq!(obj.get("y").unwrap().as_integer(), Some(2));
    }

    // -- YAML parser --------------------------------------------------------

    #[test]
    fn test_yaml_basic() {
        let yaml = "name: omega\nport: 8080\ndebug: true\n";
        let cfg = YamlParser::parse(yaml).unwrap();
        assert_eq!(cfg.get_string("name"), Some("omega"));
        assert_eq!(cfg.get_integer("port"), Some(8080));
        assert_eq!(cfg.get_bool("debug"), Some(true));
    }

    #[test]
    fn test_yaml_nested() {
        let yaml = "server:\n  host: localhost\n  port: 443\n";
        let cfg = YamlParser::parse(yaml).unwrap();
        assert_eq!(cfg.get_string("server.host"), Some("localhost"));
        assert_eq!(cfg.get_integer("server.port"), Some(443));
    }

    // -- Env loader ---------------------------------------------------------

    #[test]
    fn test_env_loader() {
        let pairs = vec![
            ("APP_DB_HOST", "localhost"),
            ("APP_DB_PORT", "5432"),
            ("APP_DEBUG", "true"),
            ("OTHER_KEY", "ignored"),
        ];
        let loader = EnvLoader::new("APP");
        let cfg = loader.load_from_pairs(&pairs);
        assert_eq!(cfg.get_string("db.host"), Some("localhost"));
        assert_eq!(cfg.get_integer("db.port"), Some(5432));
        assert_eq!(cfg.get_bool("debug"), Some(true));
        assert!(!cfg.has("other.key"));
    }

    #[test]
    fn test_env_custom_separator() {
        let pairs = vec![("MY__X", "1")];
        let loader = EnvLoader::new("MY").with_separator("__");
        let cfg = loader.load_from_pairs(&pairs);
        assert_eq!(cfg.get_integer("x"), Some(1));
    }

    // -- Validation ---------------------------------------------------------

    #[test]
    fn test_validation_passes() {
        let mut cfg = Config::new();
        cfg.set("host", ConfigValue::String("0.0.0.0".into()));
        cfg.set("port", ConfigValue::Integer(8080));

        let v = ConfigValidator::new()
            .require("host", ValueKind::String)
            .require("port", ValueKind::Integer)
            .optional("debug", ValueKind::Boolean);

        assert!(v.validate(&cfg).is_ok());
    }

    #[test]
    fn test_validation_fails_missing() {
        let cfg = Config::new();
        let v = ConfigValidator::new().require("host", ValueKind::String);
        let errs = v.validate(&cfg).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("missing"));
    }

    #[test]
    fn test_validation_fails_wrong_type() {
        let mut cfg = Config::new();
        cfg.set("port", ConfigValue::String("oops".into()));
        let v = ConfigValidator::new().require("port", ValueKind::Integer);
        let errs = v.validate(&cfg).unwrap_err();
        assert!(errs[0].contains("wrong type"));
    }

    // -- Deep merge ---------------------------------------------------------

    #[test]
    fn test_deep_merge_objects() {
        let mut base = Config::new();
        base.set(
            "server",
            ConfigValue::Object({
                let mut m = HashMap::new();
                m.insert("host".into(), ConfigValue::String("a".into()));
                m.insert("port".into(), ConfigValue::Integer(80));
                m
            }),
        );

        let mut overlay = Config::new();
        overlay.set(
            "server",
            ConfigValue::Object({
                let mut m = HashMap::new();
                m.insert("port".into(), ConfigValue::Integer(443));
                m.insert("tls".into(), ConfigValue::Boolean(true));
                m
            }),
        );

        deep_merge(&mut base, &overlay);

        let server = base.get("server").unwrap().as_object().unwrap();
        assert_eq!(server.get("host").unwrap().as_string(), Some("a"));
        assert_eq!(server.get("port").unwrap().as_integer(), Some(443));
        assert_eq!(server.get("tls").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn test_deep_merge_scalar_override() {
        let mut base = Config::new();
        base.set("x", ConfigValue::Integer(1));

        let mut overlay = Config::new();
        overlay.set("x", ConfigValue::Integer(2));

        deep_merge(&mut base, &overlay);
        assert_eq!(base.get_integer("x"), Some(2));
    }
}
