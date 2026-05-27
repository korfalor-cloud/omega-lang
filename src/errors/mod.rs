use thiserror::Error;
use std::fmt;

pub type OmegaResult<T> = Result<T, OmegaError>;

#[derive(Error, Debug, Clone)]
pub enum OmegaError {
    #[error("Syntax error at line {line}, col {col}: {message}")]
    SyntaxError { line: usize, col: usize, message: String },

    #[error("Type error: {message}")]
    TypeError { message: String, span: Option<Span> },

    #[error("Name error: '{name}' is not defined")]
    NameError { name: String, span: Option<Span> },

    #[error("Runtime error: {message}")]
    RuntimeError { message: String, span: Option<Span> },

    #[error("Memory error: {message}")]
    MemoryError { message: String },

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Parse error at {location}: {message}")]
    ParseError { location: String, message: String },

    #[error("Compilation error: {message}")]
    CompilationError { message: String, span: Option<Span> },

    #[error("Stack overflow")]
    StackOverflow,

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Index out of bounds: index {index}, length {length}")]
    IndexOutOfBounds { index: i64, length: usize },

    #[error("Key error: '{key}' not found")]
    KeyError { key: String },

    #[error("Attribute error: '{attr}' not found on {ty}")]
    AttributeError { attr: String, ty: String },

    #[error("Import error: module '{module}' not found")]
    ImportError { module: String },

    #[error("Assertion error: {message}")]
    AssertionError { message: String },

    #[error("Value error: {message}")]
    ValueError { message: String },

    #[error("Overflow error: {message}")]
    OverflowError { message: String },

    #[error("Stop iteration")]
    StopIteration,

    #[error("Not implemented: {feature}")]
    NotImplementedError { feature: String },

    #[error("Permission error: {message}")]
    PermissionError { message: String },

    #[error("Timeout error: operation timed out after {seconds}s")]
    TimeoutError { seconds: f64 },

    #[error("Package error: {message}")]
    PackageError { message: String },

    #[error("Format error: {message}")]
    FormatError { message: String },

    #[error("Encoding error: {message}")]
    EncodingError { message: String },

    #[error("Thread error: {message}")]
    ThreadError { message: String },

    #[error("Channel error: {message}")]
    ChannelError { message: String },

    #[error("Lock error: {message}")]
    LockError { message: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Regex error: {message}")]
    RegexError { message: String },

    #[error("Internal error: {message}")]
    InternalError { message: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: Position,
    pub end: Position,
    pub file_id: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Position {
    pub line: usize,
    pub col: usize,
    pub offset: usize,
}

impl Position {
    pub fn new(line: usize, col: usize, offset: usize) -> Self {
        Self { line, col, offset }
    }

    pub fn advance(&mut self, ch: char) {
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        self.offset += ch.len_utf8();
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

impl Span {
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end, file_id: None }
    }

    pub fn with_file(start: Position, end: Position, file_id: usize) -> Self {
        Self { start, end, file_id: Some(file_id) }
    }

    pub fn merge(&self, other: &Span) -> Span {
        let start = if self.start.offset < other.start.offset {
            self.start.clone()
        } else {
            other.start.clone()
        };
        let end = if self.end.offset > other.end.offset {
            self.end.clone()
        } else {
            other.end.clone()
        };
        Span { start, end, file_id: self.file_id }
    }

    pub fn length(&self) -> usize {
        self.end.offset - self.start.offset
    }

    pub fn contains(&self, offset: usize) -> bool {
        offset >= self.start.offset && offset < self.end.offset
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.start, self.end)
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub span: Option<Span>,
    pub notes: Vec<String>,
    pub help: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Info,
    Hint,
}

impl fmt::Display for DiagnosticLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticLevel::Error => write!(f, "error"),
            DiagnosticLevel::Warning => write!(f, "warning"),
            DiagnosticLevel::Info => write!(f, "info"),
            DiagnosticLevel::Hint => write!(f, "hint"),
        }
    }
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Error,
            message: message.into(),
            span: None,
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Warning,
            message: message.into(),
            span: None,
            notes: Vec::new(),
            help: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct DiagnosticBag {
    pub diagnostics: Vec<Diagnostic>,
    pub source: String,
    pub filename: String,
}

impl DiagnosticBag {
    pub fn new(source: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            diagnostics: Vec::new(),
            source: source.into(),
            filename: filename.into(),
        }
    }

    pub fn report(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.level == DiagnosticLevel::Error)
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Error).count()
    }

    pub fn warning_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Warning).count()
    }

    pub fn report_syntax_error(&mut self, line: usize, col: usize, message: impl Into<String>) {
        let pos = Position::new(line, col, 0);
        let span = Span::new(pos.clone(), Position::new(line, col + 1, 0));
        self.report(Diagnostic::error(message).with_span(span));
    }

    pub fn report_type_error(&mut self, message: impl Into<String>, span: Option<Span>) {
        self.report(Diagnostic::error(message).with_span(span.unwrap_or_else(|| {
            let pos = Position::new(1, 1, 0);
            Span::new(pos.clone(), pos)
        })));
    }

    pub fn report_name_error(&mut self, name: &str, span: Option<Span>) {
        self.report(
            Diagnostic::error(format!("'{}' is not defined", name))
                .with_span(span.unwrap_or_else(|| {
                    let pos = Position::new(1, 1, 0);
                    Span::new(pos.clone(), pos)
                }))
                .with_help("Did you forget to declare this variable?")
        );
    }

    pub fn format_diagnostics(&self) -> String {
        let mut output = String::new();
        for diag in &self.diagnostics {
            output.push_str(&format!("{}: {}\n", diag.level, diag.message));
            if let Some(span) = &diag.span {
                output.push_str(&format!("  --> {}:{}\n", self.filename, span.start));
                if let Some(line_text) = self.get_line(span.start.line) {
                    output.push_str(&format!("   |\n"));
                    output.push_str(&format!("{:4} | {}\n", span.start.line, line_text));
                    output.push_str(&format!("   | {}{}\n",
                        " ".repeat(span.start.col.saturating_sub(1)),
                        "^".repeat(span.length().max(1))
                    ));
                }
            }
            for note in &diag.notes {
                output.push_str(&format!("  = note: {}\n", note));
            }
            if let Some(help) = &diag.help {
                output.push_str(&format!("  = help: {}\n", help));
            }
            output.push('\n');
        }
        output
    }

    fn get_line(&self, line_num: usize) -> Option<&str> {
        self.source.lines().nth(line_num.saturating_sub(1))
    }
}
