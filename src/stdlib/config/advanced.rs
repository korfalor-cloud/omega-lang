/// Advanced configuration: TOML/YAML parsers, env loader, validation, deep merge.
use std::collections::HashMap;
use super::config::{Config, ConfigValue};

/// TOML parser with sections, inline arrays, and inline tables.
pub struct TomlParser;
impl TomlParser {
    pub fn parse(input: &str) -> Result<Config, String> {
        let mut config = Config::new();
        let mut section = String::new();
        for (ln, raw) in input.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if line.starts_with('[') && line.ends_with(']') {
                section = line[1..line.len()-1].trim().to_string(); continue;
            }
            let eq = line.find('=').ok_or_else(|| format!("line {}: expected '='", ln+1))?;
            let key = line[..eq].trim();
            let full = if section.is_empty() { key.to_string() } else { format!("{}.{}", section, key) };
            config.set(&full, Self::val(line[eq+1..].trim())?);
        }
        Ok(config)
    }
    fn val(raw: &str) -> Result<ConfigValue, String> {
        if raw.starts_with('[') && raw.ends_with(']') {
            return Ok(ConfigValue::Array(raw[1..raw.len()-1].split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| Self::val(s.trim())).collect::<Result<_,_>>()?));
        }
        if raw.starts_with('{') && raw.ends_with('}') {
            let mut m = HashMap::new();
            for p in raw[1..raw.len()-1].split(',') {
                let p = p.trim(); if p.is_empty() { continue; }
                let e = p.find('=').ok_or("expected '=' in inline table")?;
                m.insert(p[..e].trim().trim_matches('"').to_string(), Self::val(p[e+1..].trim())?);
            }
            return Ok(ConfigValue::Object(m));
        }
        if raw.starts_with('"') && raw.ends_with('"') { return Ok(ConfigValue::String(raw[1..raw.len()-1].to_string())); }
        if raw == "true" { return Ok(ConfigValue::Boolean(true)); }
        if raw == "false" { return Ok(ConfigValue::Boolean(false)); }
        if let Ok(n) = raw.parse::<i64>() { return Ok(ConfigValue::Integer(n)); }
        if let Ok(f) = raw.parse::<f64>() { return Ok(ConfigValue::Float(f)); }
        Ok(ConfigValue::String(raw.to_string()))
    }
}

/// Minimal YAML parser: key-value pairs, nested maps, lists.
pub struct YamlParser;
impl YamlParser {
    pub fn parse(input: &str) -> Result<Config, String> {
        let mut config = Config::new();
        let lines: Vec<&str> = input.lines().collect();
        Self::block(&lines, 0, &mut config, "")?;
        Ok(config)
    }
    fn block(lines: &[&str], start: usize, cfg: &mut Config, pre: &str) -> Result<usize, String> {
        let mut i = start;
        while i < lines.len() {
            let raw = lines[i];
            let indent = raw.chars().take_while(|c| *c == ' ').count();
            let t = raw.trim();
            if t.is_empty() || t.starts_with('#') { i += 1; continue; }
            if let Some(colon) = t.find(':') {
                let key = t[..colon].trim();
                let vp = t[colon+1..].trim();
                let full = if pre.is_empty() { key.to_string() } else { format!("{}.{}", pre, key) };
                if vp.is_empty() {
                    i = Self::block(lines, i + 1, cfg, &full)?;
                } else {
                    cfg.set(&full, Self::scalar(vp)); i += 1;
                }
            } else if t.starts_with('-') {
                let mut idx = 0;
                while cfg.has(&format!("{}[{}]", pre, idx)) { idx += 1; }
                cfg.set(&format!("{}[{}]", pre, idx), Self::scalar(t[1..].trim()));
                i += 1;
            } else { i += 1; }
            if i < lines.len() {
                let ni = lines[i].chars().take_while(|c| *c == ' ').count();
                let nt = lines[i].trim();
                if !nt.is_empty() && !nt.starts_with('#') && ni < indent { break; }
            }
        }
        Ok(i)
    }
    fn scalar(raw: &str) -> ConfigValue {
        if (raw.starts_with('"') && raw.ends_with('"')) || (raw.starts_with('\'') && raw.ends_with('\''))
        { return ConfigValue::String(raw[1..raw.len()-1].to_string()); }
        if raw == "true" { return ConfigValue::Boolean(true); }
        if raw == "false" { return ConfigValue::Boolean(false); }
        if raw == "null" || raw == "~" { return ConfigValue::String(String::new()); }
        if let Ok(n) = raw.parse::<i64>() { return ConfigValue::Integer(n); }
        if let Ok(f) = raw.parse::<f64>() { return ConfigValue::Float(f); }
        ConfigValue::String(raw.to_string())
    }
}

/// Loads config from environment-style key-value pairs.
pub struct EnvLoader { prefix: String, separator: String }
impl EnvLoader {
    pub fn new(prefix: &str) -> Self { Self { prefix: prefix.into(), separator: "_".into() } }
    pub fn with_separator(mut self, sep: &str) -> Self { self.separator = sep.into(); self }
    pub fn load_from_pairs(&self, pairs: &[(&str, &str)]) -> Config {
        let mut config = Config::new();
        let pfx = format!("{}{}", self.prefix, self.separator);
        for (k, v) in pairs {
            if let Some(suffix) = k.strip_prefix(&pfx) {
                let key = suffix.to_lowercase().replace(&self.separator, ".");
                if let Ok(n) = v.parse::<i64>() { config.set(&key, ConfigValue::Integer(n)); }
                else if let Ok(f) = v.parse::<f64>() { config.set(&key, ConfigValue::Float(f)); }
                else if *v == "true" || *v == "false" { config.set(&key, ConfigValue::Boolean(*v == "true")); }
                else { config.set(&key, ConfigValue::String(v.to_string())); }
            }
        }
        config
    }
}

/// Config validation builder.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueKind { String, Integer, Float, Boolean, Any }
pub struct ConfigValidator { rules: Vec<(String, bool, ValueKind)> }
impl ConfigValidator {
    pub fn new() -> Self { Self { rules: Vec::new() } }
    pub fn require(mut self, key: &str, kind: ValueKind) -> Self { self.rules.push((key.into(), true, kind)); self }
    pub fn optional(mut self, key: &str, kind: ValueKind) -> Self { self.rules.push((key.into(), false, kind)); self }
    pub fn validate(&self, config: &Config) -> Result<(), Vec<String>> {
        let mut errs = Vec::new();
        for (key, req, kind) in &self.rules {
            match config.get(key) {
                None => if *req { errs.push(format!("missing required key '{}'", key)); },
                Some(v) => {
                    let ok = matches!((kind, v),
                        (ValueKind::Any, _) | (ValueKind::String, ConfigValue::String(_))
                        | (ValueKind::Integer, ConfigValue::Integer(_))
                        | (ValueKind::Float, ConfigValue::Float(_) | ConfigValue::Integer(_))
                        | (ValueKind::Boolean, ConfigValue::Boolean(_)));
                    if !ok { errs.push(format!("key '{}' has wrong type (expected {:?})", key, kind)); }
                }
            }
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

/// Deep-merge `overlay` into `base`. Nested objects are merged recursively.
pub fn deep_merge(base: &mut Config, overlay: &Config) {
    for key in overlay.keys() {
        let k = key.to_string();
        match (base.get(&k), overlay.get(&k)) {
            (Some(ConfigValue::Object(b)), Some(ConfigValue::Object(o))) => {
                let mut merged = b.clone();
                for (ik, iv) in o { merged.insert(ik.clone(), iv.clone()); }
                base.set(&k, ConfigValue::Object(merged));
            }
            (_, Some(val)) => { base.set(&k, val.clone()); }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml_basic() {
        let c = TomlParser::parse("# c\nname = \"omega\"\nport = 8080\ndebug = true\nratio = 3.14\n").unwrap();
        assert_eq!(c.get_string("name"), Some("omega"));
        assert_eq!(c.get_integer("port"), Some(8080));
        assert_eq!(c.get_bool("debug"), Some(true));
        assert!((c.get_float("ratio").unwrap() - 3.14).abs() < f64::EPSILON);
    }
    #[test]
    fn test_toml_sections() {
        let c = TomlParser::parse("[server]\nhost = \"0.0.0.0\"\nport = 443\n").unwrap();
        assert_eq!(c.get_string("server.host"), Some("0.0.0.0"));
        assert_eq!(c.get_integer("server.port"), Some(443));
    }
    #[test]
    fn test_toml_inline_array() {
        let c = TomlParser::parse("ports = [80, 443, 8080]\n").unwrap();
        let a = c.get("ports").unwrap().as_array().unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a[0].as_integer(), Some(80));
    }
    #[test]
    fn test_toml_inline_table() {
        let c = TomlParser::parse("point = { x = 1, y = 2 }\n").unwrap();
        let o = c.get("point").unwrap().as_object().unwrap();
        assert_eq!(o.get("x").unwrap().as_integer(), Some(1));
    }
    #[test]
    fn test_yaml_basic() {
        let c = YamlParser::parse("name: omega\nport: 8080\ndebug: true\n").unwrap();
        assert_eq!(c.get_string("name"), Some("omega"));
        assert_eq!(c.get_integer("port"), Some(8080));
        assert_eq!(c.get_bool("debug"), Some(true));
    }
    #[test]
    fn test_yaml_nested() {
        let c = YamlParser::parse("server:\n  host: localhost\n  port: 443\n").unwrap();
        assert_eq!(c.get_string("server.host"), Some("localhost"));
        assert_eq!(c.get_integer("server.port"), Some(443));
    }
    #[test]
    fn test_env_loader() {
        let c = EnvLoader::new("APP").load_from_pairs(&[
            ("APP_DB_HOST", "localhost"), ("APP_DB_PORT", "5432"),
            ("APP_DEBUG", "true"), ("OTHER_KEY", "ignored"),
        ]);
        assert_eq!(c.get_string("db.host"), Some("localhost"));
        assert_eq!(c.get_integer("db.port"), Some(5432));
        assert_eq!(c.get_bool("debug"), Some(true));
        assert!(!c.has("other.key"));
    }
    #[test]
    fn test_env_custom_sep() {
        let c = EnvLoader::new("MY").with_separator("__").load_from_pairs(&[("MY__X", "1")]);
        assert_eq!(c.get_integer("x"), Some(1));
    }
    #[test]
    fn test_validation_passes() {
        let mut c = Config::new();
        c.set("host", ConfigValue::String("0.0.0.0".into()));
        c.set("port", ConfigValue::Integer(8080));
        assert!(ConfigValidator::new()
            .require("host", ValueKind::String).require("port", ValueKind::Integer)
            .optional("debug", ValueKind::Boolean).validate(&c).is_ok());
    }
    #[test]
    fn test_validation_fails_missing() {
        let e = ConfigValidator::new().require("host", ValueKind::String)
            .validate(&Config::new()).unwrap_err();
        assert_eq!(e.len(), 1);
        assert!(e[0].contains("missing"));
    }
    #[test]
    fn test_validation_fails_type() {
        let mut c = Config::new();
        c.set("port", ConfigValue::String("oops".into()));
        let e = ConfigValidator::new().require("port", ValueKind::Integer).validate(&c).unwrap_err();
        assert!(e[0].contains("wrong type"));
    }
    #[test]
    fn test_deep_merge_objects() {
        let mut base = Config::new();
        let mut bm = HashMap::new();
        bm.insert("host".into(), ConfigValue::String("a".into()));
        bm.insert("port".into(), ConfigValue::Integer(80));
        base.set("server", ConfigValue::Object(bm));
        let mut ov = Config::new();
        let mut om = HashMap::new();
        om.insert("port".into(), ConfigValue::Integer(443));
        om.insert("tls".into(), ConfigValue::Boolean(true));
        ov.set("server", ConfigValue::Object(om));
        deep_merge(&mut base, &ov);
        let s = base.get("server").unwrap().as_object().unwrap();
        assert_eq!(s.get("host").unwrap().as_string(), Some("a"));
        assert_eq!(s.get("port").unwrap().as_integer(), Some(443));
        assert_eq!(s.get("tls").unwrap().as_bool(), Some(true));
    }
    #[test]
    fn test_deep_merge_scalar() {
        let mut b = Config::new(); b.set("x", ConfigValue::Integer(1));
        let mut o = Config::new(); o.set("x", ConfigValue::Integer(2));
        deep_merge(&mut b, &o);
        assert_eq!(b.get_integer("x"), Some(2));
    }
}
