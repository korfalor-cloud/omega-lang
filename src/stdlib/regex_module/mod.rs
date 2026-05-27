use regex::Regex;
use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;

pub struct OmegaRegex {
    inner: Regex,
}

impl OmegaRegex {
    pub fn new(pattern: &str) -> OmegaResult<Self> {
        let inner = Regex::new(pattern).map_err(|e| OmegaError::RegexError {
            message: e.to_string(),
        })?;
        Ok(Self { inner })
    }

    pub fn is_match(&self, text: &str) -> bool {
        self.inner.is_match(text)
    }

    pub fn find(&self, text: &str) -> Option<String> {
        self.inner.find(text).map(|m| m.as_str().to_string())
    }

    pub fn find_all(&self, text: &str) -> Vec<String> {
        self.inner.find_iter(text).map(|m| m.as_str().to_string()).collect()
    }

    pub fn capture(&self, text: &str) -> Option<Vec<String>> {
        self.inner.captures(text).map(|caps| {
            caps.iter()
                .map(|m| m.map(|m| m.as_str().to_string()).unwrap_or_default())
                .collect()
        })
    }

    pub fn capture_all(&self, text: &str) -> Vec<Vec<String>> {
        self.inner.captures_iter(text)
            .map(|caps| {
                caps.iter()
                    .map(|m| m.map(|m| m.as_str().to_string()).unwrap_or_default())
                    .collect()
            })
            .collect()
    }

    pub fn replace(&self, text: &str, replacement: &str) -> String {
        self.inner.replace(text, replacement).to_string()
    }

    pub fn replace_all(&self, text: &str, replacement: &str) -> String {
        self.inner.replace_all(text, replacement).to_string()
    }

    pub fn replace_with(&self, text: &str, f: impl Fn(&str) -> String) -> String {
        let mut result = String::new();
        let mut last_end = 0;
        for cap in self.inner.captures_iter(text) {
            let m = cap.get(0).unwrap();
            result.push_str(&text[last_end..m.start()]);
            result.push_str(&f(m.as_str()));
            last_end = m.end();
        }
        result.push_str(&text[last_end..]);
        result
    }

    pub fn split(&self, text: &str) -> Vec<String> {
        self.inner.split(text).map(String::from).collect()
    }

    pub fn pattern(&self) -> &str {
        self.inner.as_str()
    }

    pub fn group_names(&self) -> Vec<String> {
        self.inner.capture_names()
            .filter_map(|n| n.map(String::from))
            .collect()
    }

    pub fn group_count(&self) -> usize {
        self.inner.captures_len()
    }
}

pub fn is_valid_pattern(pattern: &str) -> bool {
    Regex::new(pattern).is_ok()
}

pub fn escape_pattern(s: &str) -> String {
    regex::escape(s)
}
