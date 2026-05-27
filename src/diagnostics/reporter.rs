use crate::errors::{Diagnostic, DiagnosticLevel, Span};
use colored::*;

pub struct DiagnosticReporter {
    source: String,
    filename: String,
}

impl DiagnosticReporter {
    pub fn new(source: &str, filename: &str) -> Self {
        Self {
            source: source.to_string(),
            filename: filename.to_string(),
        }
    }

    pub fn report(&self, diagnostic: &Diagnostic) -> String {
        let level = match diagnostic.level {
            DiagnosticLevel::Error => "error".red().bold(),
            DiagnosticLevel::Warning => "warning".yellow().bold(),
            DiagnosticLevel::Info => "info".blue().bold(),
            DiagnosticLevel::Hint => "hint".cyan().bold(),
        };

        let mut output = format!("{}: {}\n", level, diagnostic.message);

        if let Some(span) = &diagnostic.span {
            output.push_str(&format!("  {} {}:{}\n", "-->".blue(), self.filename, span.start));

            if let Some(line_text) = self.get_line(span.start.line) {
                output.push_str(&format!("   {}\n", "|".blue()));
                output.push_str(&format!("{} {} {}\n",
                    format!("{:>4}", span.start.line).blue(),
                    "|".blue(),
                    line_text
                ));
                output.push_str(&format!("   {} {}{}\n",
                    "|".blue(),
                    " ".repeat(span.start.col.saturating_sub(1)),
                    "^".repeat(span.length().max(1)).red()
                ));
            }
        }

        for note in &diagnostic.notes {
            output.push_str(&format!("  {} {}\n", "= note:".blue(), note));
        }

        if let Some(help) = &diagnostic.help {
            output.push_str(&format!("  {} {}\n", "= help:".cyan(), help));
        }

        output
    }

    pub fn report_all(&self, diagnostics: &[Diagnostic]) -> String {
        diagnostics.iter().map(|d| self.report(d)).collect::<Vec<_>>().join("\n")
    }

    pub fn format_summary(&self, diagnostics: &[Diagnostic]) -> String {
        let errors = diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Error).count();
        let warnings = diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Warning).count();

        if errors == 0 && warnings == 0 {
            return String::new();
        }

        let mut parts = Vec::new();
        if errors > 0 {
            parts.push(format!("{} error{}", errors, if errors == 1 { "" } else { "s" }));
        }
        if warnings > 0 {
            parts.push(format!("{} warning{}", warnings, if warnings == 1 { "" } else { "s" }));
        }

        format!("aborting due to {}", parts.join("; "))
    }

    fn get_line(&self, line_num: usize) -> Option<&str> {
        self.source.lines().nth(line_num.saturating_sub(1))
    }
}
