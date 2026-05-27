/// Environment variable configuration.

use std::collections::HashMap;

#[derive(Debug)]
pub struct EnvConfig {
    prefix: String,
    separator: String,
    values: HashMap<String, String>,
}

impl EnvConfig {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
            separator: "_".to_string(),
            values: HashMap::new(),
        }
    }

    pub fn with_separator(mut self, sep: &str) -> Self {
        self.separator = sep.to_string();
        self
    }

    pub fn load(&mut self) {
        // In a real implementation, would read from std::env
        // For now, just store the structure
    }

    pub fn set(&mut self, key: &str, value: &str) {
        let full_key = format!("{}{}{}", self.prefix, self.separator, key);
        self.values.insert(full_key.to_uppercase(), value.to_string());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        let full_key = format!("{}{}{}", self.prefix, self.separator, key);
        self.values.get(&full_key.to_uppercase()).map(|s| s.as_str())
    }

    pub fn get_or_default(&self, key: &str, default: &str) -> String {
        self.get(key).unwrap_or(default).to_string()
    }

    pub fn get_integer(&self, key: &str) -> Option<i64> {
        self.get(key)?.parse().ok()
    }

    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.get(key)?.parse().ok()
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            "true" | "1" | "yes" => Some(true),
            "false" | "0" | "no" => Some(false),
            _ => None,
        }
    }

    pub fn keys(&self) -> Vec<&str> {
        self.values.keys().map(|s| s.as_str()).collect()
    }

    pub fn to_map(&self) -> &HashMap<String, String> {
        &self.values
    }
}

/// CLI argument parser
#[derive(Debug)]
pub struct CliArgs {
    args: Vec<String>,
    parsed: HashMap<String, String>,
    flags: Vec<String>,
}

impl CliArgs {
    pub fn new(args: Vec<String>) -> Self {
        Self {
            args,
            parsed: HashMap::new(),
            flags: Vec::new(),
        }
    }

    pub fn parse(&mut self) {
        let mut i = 0;
        while i < self.args.len() {
            let arg = &self.args[i];
            if arg.starts_with("--") {
                let key = arg[2..].to_string();
                if i + 1 < self.args.len() && !self.args[i + 1].starts_with('-') {
                    self.parsed.insert(key, self.args[i + 1].clone());
                    i += 2;
                } else {
                    self.flags.push(key);
                    i += 1;
                }
            } else if arg.starts_with('-') {
                let key = arg[1..].to_string();
                if i + 1 < self.args.len() && !self.args[i + 1].starts_with('-') {
                    self.parsed.insert(key, self.args[i + 1].clone());
                    i += 2;
                } else {
                    self.flags.push(key);
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.parsed.get(key).map(|s| s.as_str())
    }

    pub fn get_or_default(&self, key: &str, default: &str) -> String {
        self.get(key).unwrap_or(default).to_string()
    }

    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.contains(&flag.to_string())
    }

    pub fn get_integer(&self, key: &str) -> Option<i64> {
        self.get(key)?.parse().ok()
    }

    pub fn positional(&self) -> Vec<&str> {
        self.args.iter()
            .filter(|a| !a.starts_with('-'))
            .map(|s| s.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_config() {
        let mut env = EnvConfig::new("APP");
        env.set("DB_HOST", "localhost");
        env.set("DB_PORT", "5432");

        assert_eq!(env.get("DB_HOST"), Some("localhost"));
        assert_eq!(env.get_integer("DB_PORT"), Some(5432));
    }

    #[test]
    fn test_env_bool() {
        let mut env = EnvConfig::new("APP");
        env.set("DEBUG", "true");
        env.set("VERBOSE", "0");

        assert_eq!(env.get_bool("DEBUG"), Some(true));
        assert_eq!(env.get_bool("VERBOSE"), Some(false));
    }

    #[test]
    fn test_cli_args() {
        let args = vec![
            "program".to_string(),
            "--port".to_string(),
            "8080".to_string(),
            "--verbose".to_string(),
            "file.txt".to_string(),
        ];

        let mut cli = CliArgs::new(args);
        cli.parse();

        assert_eq!(cli.get("port"), Some("8080"));
        assert!(cli.has_flag("verbose"));
        assert_eq!(cli.positional(), vec!["file.txt"]);
    }
}
