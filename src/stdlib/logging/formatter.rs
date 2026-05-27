/// Log formatting utilities.

use super::logger::{LogEntry, LogLevel};

pub trait LogFormatter {
    fn format(&self, entry: &LogEntry) -> String;
}

pub struct JsonFormatter;
pub struct TextFormatter;
pub struct CompactFormatter;
pub struct StructuredFormatter;

impl LogFormatter for JsonFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let fields: String = entry.fields.iter()
            .map(|(k, v)| format!("\"{}\":\"{}\"", k, v))
            .collect::<Vec<_>>()
            .join(",");

        format!(
            r#"{{"level":"{}","message":"{}","timestamp":{},"fields":{{{}}}}}"#,
            entry.level.as_str(),
            entry.message.replace('"', "\\\""),
            entry.timestamp,
            fields
        )
    }
}

impl LogFormatter for TextFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let fields: String = if entry.fields.is_empty() {
            String::new()
        } else {
            let f: String = entry.fields.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(" ");
            format!(" {}", f)
        };

        format!(
            "[{}] {} {}{}",
            entry.level.as_str(),
            entry.timestamp,
            entry.message,
            fields
        )
    }
}

impl LogFormatter for CompactFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let level_char = match entry.level {
            LogLevel::Trace => "T",
            LogLevel::Debug => "D",
            LogLevel::Info => "I",
            LogLevel::Warn => "W",
            LogLevel::Error => "E",
            LogLevel::Fatal => "F",
        };
        format!("{} {}", level_char, entry.message)
    }
}

impl LogFormatter for StructuredFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let mut parts = vec![
            format!("level={}", entry.level.as_str()),
            format!("msg=\"{}\"", entry.message),
            format!("ts={}", entry.timestamp),
        ];

        for (k, v) in &entry.fields {
            parts.push(format!("{}={}", k, v));
        }

        parts.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_entry() -> LogEntry {
        LogEntry {
            level: LogLevel::Info,
            message: "test message".to_string(),
            timestamp: 1234567890,
            fields: HashMap::new(),
            module: None,
            line: None,
        }
    }

    #[test]
    fn test_json_formatter() {
        let formatter = JsonFormatter;
        let output = formatter.format(&sample_entry());
        assert!(output.contains("\"level\":\"INFO\""));
        assert!(output.contains("\"message\":\"test message\""));
    }

    #[test]
    fn test_text_formatter() {
        let formatter = TextFormatter;
        let output = formatter.format(&sample_entry());
        assert!(output.contains("[INFO]"));
        assert!(output.contains("test message"));
    }

    #[test]
    fn test_compact_formatter() {
        let formatter = CompactFormatter;
        let output = formatter.format(&sample_entry());
        assert_eq!(output, "I test message");
    }

    #[test]
    fn test_structured_formatter() {
        let formatter = StructuredFormatter;
        let output = formatter.format(&sample_entry());
        assert!(output.contains("level=INFO"));
        assert!(output.contains("msg=\"test message\""));
    }
}
