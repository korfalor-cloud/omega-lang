use std::collections::HashMap;
use crate::ast::*;
use crate::errors::{Diagnostic, DiagnosticSeverity, Span};
use super::rules::*;

pub struct Linter {
    rules: Vec<Box<dyn LintRule>>,
    config: LintConfig,
    diagnostics: Vec<LintDiagnostic>,
}

#[derive(Debug, Clone)]
pub struct LintConfig {
    pub max_line_length: usize,
    pub max_function_length: usize,
    pub max_complexity: usize,
    pub max_params: usize,
    pub max_depth: usize,
    pub allow_underscore: bool,
    pub allow_mutable_globals: bool,
    pub strict_types: bool,
    pub naming_convention: NamingConvention,
}

#[derive(Debug, Clone)]
pub enum NamingConvention {
    SnakeCase,
    CamelCase,
    PascalCase,
    Mixed,
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            max_line_length: 100,
            max_function_length: 50,
            max_complexity: 10,
            max_params: 5,
            max_depth: 4,
            allow_underscore: true,
            allow_mutable_globals: false,
            strict_types: true,
            naming_convention: NamingConvention::SnakeCase,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LintDiagnostic {
    pub rule: String,
    pub message: String,
    pub severity: LintSeverity,
    pub line: usize,
    pub column: usize,
    pub end_line: usize,
    pub end_column: usize,
    pub fix: Option<LintFix>,
}

#[derive(Debug, Clone)]
pub enum LintSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

#[derive(Debug, Clone)]
pub struct LintFix {
    pub description: String,
    pub replacement: String,
}

impl Linter {
    pub fn new() -> Self {
        Self {
            rules: Self::default_rules(),
            config: LintConfig::default(),
            diagnostics: Vec::new(),
        }
    }

    pub fn with_config(config: LintConfig) -> Self {
        Self {
            rules: Self::default_rules(),
            config,
            diagnostics: Vec::new(),
        }
    }

    fn default_rules() -> Vec<Box<dyn LintRule>> {
        vec![
            Box::new(NoUnusedVariables),
            Box::new(NoShadowing),
            Box::new(NoMutableGlobals),
            Box::new(PreferConst),
            Box::new(NoMagicNumbers),
            Box::new(MaxLineLength),
            Box::new(MaxFunctionLength),
            Box::new(MaxComplexity),
            Box::new(MaxParams),
            Box::new(NamingConventionRule),
            Box::new(NoEmptyBlocks),
            Box::new(NoUnreachableCode),
            Box::new(PreferEarlyReturn),
            Box::new(NoNestedIf),
            Box::new(UseTypeAnnotations),
            Box::new(NoDeepNesting),
            Box::new(ConsistentReturn),
            Box::new(NoDuplicateImports),
            Box::new(PreferTemplateStrings),
            Box::new(NoExplicitBoolComparison),
        ]
    }

    pub fn add_rule(&mut self, rule: Box<dyn LintRule>) {
        self.rules.push(rule);
    }

    pub fn remove_rule(&mut self, name: &str) {
        self.rules.retain(|r| r.name() != name);
    }

    pub fn lint(&mut self, ast: &AstNode) -> Vec<LintDiagnostic> {
        self.diagnostics.clear();

        for rule in &self.rules {
            let mut context = LintContext::new(&self.config);
            rule.check(ast, &mut context);
            self.diagnostics.extend(context.diagnostics());
        }

        self.diagnostics.clone()
    }

    pub fn lint_with_rules(&mut self, ast: &AstNode, rule_names: &[&str]) -> Vec<LintDiagnostic> {
        self.diagnostics.clear();

        for rule in &self.rules {
            if rule_names.contains(&rule.name()) {
                let mut context = LintContext::new(&self.config);
                rule.check(ast, &mut context);
                self.diagnostics.extend(context.diagnostics());
            }
        }

        self.diagnostics.clone()
    }

    pub fn format_diagnostics(&self, diagnostics: &[LintDiagnostic], source: &str) -> String {
        let mut output = String::new();
        let lines: Vec<&str> = source.lines().collect();

        for diag in diagnostics {
            let severity = match diag.severity {
                LintSeverity::Error => "error",
                LintSeverity::Warning => "warning",
                LintSeverity::Info => "info",
                LintSeverity::Hint => "hint",
            };

            output.push_str(&format!(
                "{}: {} [{}]\n",
                severity, diag.message, diag.rule
            ));

            if diag.line > 0 && diag.line <= lines.len() {
                output.push_str(&format!("  --> line {}\n", diag.line));
                output.push_str(&format!("  {}\n", lines[diag.line - 1]));

                if diag.column > 0 {
                    let indent: String = " ".repeat(diag.column - 1);
                    output.push_str(&format!("  {}^\n", indent));
                }
            }

            if let Some(fix) = &diag.fix {
                output.push_str(&format!("  help: {}\n", fix.description));
            }

            output.push('\n');
        }

        output
    }
}

pub struct LintContext<'a> {
    config: &'a LintConfig,
    diagnostics: Vec<LintDiagnostic>,
    scope_stack: Vec<LintScope>,
    function_depth: usize,
    block_depth: usize,
}

#[derive(Debug)]
struct LintScope {
    variables: HashMap<String, VariableInfo>,
    used_variables: HashMap<String, bool>,
}

#[derive(Debug)]
struct VariableInfo {
    name: String,
    is_mutable: bool,
    is_used: bool,
    line: usize,
}

impl<'a> LintContext<'a> {
    pub fn new(config: &'a LintConfig) -> Self {
        Self {
            config,
            diagnostics: Vec::new(),
            scope_stack: vec![LintScope {
                variables: HashMap::new(),
                used_variables: HashMap::new(),
            }],
            function_depth: 0,
            block_depth: 0,
        }
    }

    pub fn diagnostics(&self) -> Vec<LintDiagnostic> {
        self.diagnostics.clone()
    }

    pub fn add_diagnostic(&mut self, diag: LintDiagnostic) {
        self.diagnostics.push(diag);
    }

    pub fn push_scope(&mut self) {
        self.scope_stack.push(LintScope {
            variables: HashMap::new(),
            used_variables: HashMap::new(),
        });
    }

    pub fn pop_scope(&mut self) {
        if let Some(scope) = self.scope_stack.pop() {
            // Check for unused variables
            for (name, info) in &scope.variables {
                if !info.is_used && !name.starts_with('_') {
                    self.diagnostics.push(LintDiagnostic {
                        rule: "no-unused-variables".to_string(),
                        message: format!("Variable '{}' is defined but never used", name),
                        severity: LintSeverity::Warning,
                        line: info.line,
                        column: 0,
                        end_line: info.line,
                        end_column: 0,
                        fix: None,
                    });
                }
            }
        }
    }

    pub fn declare_variable(&mut self, name: &str, is_mutable: bool, line: usize) {
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.variables.insert(
                name.to_string(),
                VariableInfo {
                    name: name.to_string(),
                    is_mutable,
                    is_used: false,
                    line,
                },
            );
        }
    }

    pub fn use_variable(&mut self, name: &str) {
        for scope in self.scope_stack.iter_mut().rev() {
            if let Some(info) = scope.variables.get_mut(name) {
                info.is_used = true;
                return;
            }
        }
    }

    pub fn is_variable_declared(&self, name: &str) -> bool {
        for scope in self.scope_stack.iter().rev() {
            if scope.variables.contains_key(name) {
                return true;
            }
        }
        false
    }

    pub fn config(&self) -> &LintConfig {
        self.config
    }

    pub fn enter_function(&mut self) {
        self.function_depth += 1;
    }

    pub fn exit_function(&mut self) {
        if self.function_depth > 0 {
            self.function_depth -= 1;
        }
    }

    pub fn function_depth(&self) -> usize {
        self.function_depth
    }

    pub fn enter_block(&mut self) {
        self.block_depth += 1;
    }

    pub fn exit_block(&mut self) {
        if self.block_depth > 0 {
            self.block_depth -= 1;
        }
    }

    pub fn block_depth(&self) -> usize {
        self.block_depth
    }
}

pub trait LintRule {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn check(&self, ast: &AstNode, context: &mut LintContext);
}
