use std::collections::HashMap;
use crate::ast::*;
use crate::errors::{OmegaResult, Span};
use crate::types::OmegaType;

pub struct LspServer {
    documents: HashMap<String, DocumentState>,
    capabilities: ServerCapabilities,
}

struct DocumentState {
    content: String,
    version: i32,
    ast: Option<AstNode>,
    diagnostics: Vec<Diagnostic>,
}

#[derive(Debug)]
struct Diagnostic {
    range: Range,
    severity: DiagnosticSeverity,
    message: String,
}

#[derive(Debug)]
struct Range {
    start: Position,
    end: Position,
}

#[derive(Debug)]
struct Position {
    line: u32,
    character: u32,
}

#[derive(Debug)]
enum DiagnosticSeverity {
    Error = 1,
    Warning = 2,
    Information = 3,
    Hint = 4,
}

struct ServerCapabilities {
    text_document_sync: bool,
    completion_provider: bool,
    hover_provider: bool,
    definition_provider: bool,
    references_provider: bool,
    document_symbol_provider: bool,
    workspace_symbol_provider: bool,
    code_action_provider: bool,
    formatting_provider: bool,
}

impl LspServer {
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion_provider: true,
                hover_provider: true,
                definition_provider: true,
                references_provider: true,
                document_symbol_provider: true,
                workspace_symbol_provider: true,
                code_action_provider: true,
                formatting_provider: true,
            },
        }
    }

    pub fn did_open(&mut self, uri: &str, content: &str, version: i32) {
        self.documents.insert(uri.to_string(), DocumentState {
            content: content.to_string(),
            version,
            ast: None,
            diagnostics: Vec::new(),
        });
        self.analyze(uri);
    }

    pub fn did_change(&mut self, uri: &str, content: &str, version: i32) {
        if let Some(doc) = self.documents.get_mut(uri) {
            doc.content = content.to_string();
            doc.version = version;
        }
        self.analyze(uri);
    }

    pub fn did_close(&mut self, uri: &str) {
        self.documents.remove(uri);
    }

    fn analyze(&mut self, uri: &str) {
        if let Some(doc) = self.documents.get_mut(uri) {
            let mut parser = crate::parser::Parser::new(&doc.content);
            match parser.parse() {
                Ok(ast) => {
                    doc.ast = Some(ast);
                    doc.diagnostics.clear();
                }
                Err(e) => {
                    doc.diagnostics.push(Diagnostic {
                        range: Range {
                            start: Position { line: 0, character: 0 },
                            end: Position { line: 0, character: 0 },
                        },
                        severity: DiagnosticSeverity::Error,
                        message: e.to_string(),
                    });
                }
            }
        }
    }

    pub fn completion(&self, uri: &str, line: u32, character: u32) -> Vec<CompletionItem> {
        let mut items = Vec::new();

        // Add keyword completions
        let keywords = vec![
            "let", "mut", "const", "fn", "struct", "enum", "trait", "impl",
            "if", "else", "while", "for", "loop", "match", "return", "break",
            "continue", "true", "false", "none", "async", "await", "try",
            "catch", "throw", "import", "from", "as", "pub", "mod",
        ];

        for keyword in keywords {
            items.push(CompletionItem {
                label: keyword.to_string(),
                kind: CompletionItemKind::Keyword,
                detail: None,
                documentation: None,
            });
        }

        // Add built-in function completions
        let builtins = vec![
            "print", "println", "format", "len", "type", "assert",
            "vec", "map", "set", "range", "enumerate", "zip",
        ];

        for builtin in builtins {
            items.push(CompletionItem {
                label: builtin.to_string(),
                kind: CompletionItemKind::Function,
                detail: Some("Built-in function".to_string()),
                documentation: None,
            });
        }

        items
    }

    pub fn hover(&self, uri: &str, line: u32, character: u32) -> Option<String> {
        // TODO: Implement hover information
        None
    }

    pub fn goto_definition(&self, uri: &str, line: u32, character: u32) -> Option<Location> {
        // TODO: Implement goto definition
        None
    }

    pub fn find_references(&self, uri: &str, line: u32, character: u32) -> Vec<Location> {
        // TODO: Implement find references
        Vec::new()
    }

    pub fn document_symbols(&self, uri: &str) -> Vec<DocumentSymbol> {
        // TODO: Implement document symbols
        Vec::new()
    }

    pub fn formatting(&self, uri: &str) -> Option<String> {
        if let Some(doc) = self.documents.get(uri) {
            let mut formatter = crate::formatter::Formatter::new();
            if let Some(ast) = &doc.ast {
                formatter.format(ast).ok()
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionItemKind,
    pub detail: Option<String>,
    pub documentation: Option<String>,
}

pub enum CompletionItemKind {
    Text = 1,
    Method = 2,
    Function = 3,
    Constructor = 4,
    Field = 5,
    Variable = 6,
    Class = 7,
    Interface = 8,
    Module = 9,
    Property = 10,
    Unit = 11,
    Value = 12,
    Enum = 13,
    Keyword = 14,
    Snippet = 15,
    Color = 16,
    File = 17,
    Reference = 18,
}

pub struct Location {
    pub uri: String,
    pub range: Range,
}

pub struct DocumentSymbol {
    pub name: String,
    pub kind: SymbolKind,
    pub range: Range,
    pub selection_range: Range,
    pub children: Vec<DocumentSymbol>,
}

pub enum SymbolKind {
    File = 1,
    Module = 2,
    Namespace = 3,
    Package = 4,
    Class = 5,
    Method = 6,
    Property = 7,
    Field = 8,
    Constructor = 9,
    Enum = 10,
    Interface = 11,
    Function = 12,
    Variable = 13,
    Constant = 14,
    String = 15,
    Number = 16,
    Boolean = 17,
    Array = 18,
    Object = 19,
    Key = 20,
    Null = 21,
    EnumMember = 22,
    Struct = 23,
    Event = 24,
    Operator = 25,
    TypeParameter = 26,
}
