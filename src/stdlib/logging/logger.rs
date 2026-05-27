/// Structured logging system.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

impl LogLevel {
    pub fn as_str(&self) -> &str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Fatal => "FATAL",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(LogLevel::Trace),
            "DEBUG" => Some(LogLevel::Debug),
            "INFO" => Some(LogLevel::Info),
            "WARN" | "WARNING" => Some(LogLevel::Warn),
            "ERROR" => Some(LogLevel::Error),
            "FATAL" => Some(LogLevel::Fatal),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    pub timestamp: u64,
    pub fields: HashMap<String, String>,
    pub module: Option<String>,
    pub line: Option<u32>,
}

#[derive(Debug)]
pub struct Logger {
    level: LogLevel,
    entries: Vec<LogEntry>,
    max_entries: usize,
    fields: HashMap<String, String>,
}

impl Logger {
    pub fn new(level: LogLevel) -> Self {
        Self {
            level,
            entries: Vec::new(),
            max_entries: 10000,
            fields: HashMap::new(),
        }
    }

    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.insert(key.to_string(), value.to_string());
        self
    }

    pub fn set_level(&mut self, level: LogLevel) {
        self.level = level;
    }

    pub fn log(&mut self, level: LogLevel, message: &str) {
        if level < self.level {
            return;
        }

        let entry = LogEntry {
            level,
            message: message.to_string(),
            timestamp: current_timestamp(),
            fields: self.fields.clone(),
            module: None,
            line: None,
        };

        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }

        self.entries.push(entry);
    }

    pub fn trace(&mut self, message: &str) {
        self.log(LogLevel::Trace, message);
    }

    pub fn debug(&mut self, message: &str) {
        self.log(LogLevel::Debug, message);
    }

    pub fn info(&mut self, message: &str) {
        self.log(LogLevel::Info, message);
    }

    pub fn warn(&mut self, message: &str) {
        self.log(LogLevel::Warn, message);
    }

    pub fn error(&mut self, message: &str) {
        self.log(LogLevel::Error, message);
    }

    pub fn fatal(&mut self, message: &str) {
        self.log(LogLevel::Fatal, message);
    }

    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    pub fn entries_at_level(&self, level: LogLevel) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| e.level == level).collect()
    }

    pub fn recent(&self, count: usize) -> &[LogEntry] {
        let start = self.entries.len().saturating_sub(count);
        &self.entries[start..]
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Audit logger for security events
#[derive(Debug)]
pub struct AuditLogger {
    logger: Logger,
}

impl AuditLogger {
    pub fn new() -> Self {
        Self {
            logger: Logger::new(LogLevel::Info),
        }
    }

    pub fn log_auth(&mut self, user: &str, action: &str, success: bool) {
        let status = if success { "SUCCESS" } else { "FAILURE" };
        self.logger.info(&format!("AUTH {} user={} action={}", status, user, action));
    }

    pub fn log_access(&mut self, user: &str, resource: &str, action: &str) {
        self.logger.info(&format!("ACCESS user={} resource={} action={}", user, resource, action));
    }

    pub fn log_change(&mut self, user: &str, resource: &str, field: &str, old: &str, new: &str) {
        self.logger.info(&format!("CHANGE user={} resource={} field={} old={} new={}",
            user, resource, field, old, new));
    }

    pub fn entries(&self) -> &[LogEntry] {
        self.logger.entries()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger_basic() {
        let mut logger = Logger::new(LogLevel::Info);
        logger.info("test message");
        logger.debug("should not appear");

        assert_eq!(logger.count(), 1);
        assert_eq!(logger.entries()[0].message, "test message");
    }

    #[test]
    fn test_log_levels() {
        let mut logger = Logger::new(LogLevel::Trace);
        logger.trace("trace");
        logger.debug("debug");
        logger.info("info");
        logger.warn("warn");
        logger.error("error");

        assert_eq!(logger.count(), 5);
    }

    #[test]
    fn test_log_level_filter() {
        let mut logger = Logger::new(LogLevel::Warn);
        logger.info("ignored");
        logger.warn("kept");
        logger.error("kept");

        assert_eq!(logger.count(), 2);
    }

    #[test]
    fn test_logger_with_fields() {
        let mut logger = Logger::new(LogLevel::Info)
            .with_field("service", "api");
        logger.info("started");

        assert_eq!(logger.entries()[0].fields.get("service").unwrap(), "api");
    }

    #[test]
    fn test_audit_logger() {
        let mut audit = AuditLogger::new();
        audit.log_auth("alice", "login", true);
        audit.log_access("alice", "/api/users", "read");

        assert_eq!(audit.entries().len(), 2);
    }
}
