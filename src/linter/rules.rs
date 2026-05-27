use crate::ast::*;
use super::linter::{LintContext, LintDiagnostic, LintSeverity, LintFix, LintRule};

// No unused variables
pub struct NoUnusedVariables;

impl LintRule for NoUnusedVariables {
    fn name(&self) -> &str {
        "no-unused-variables"
    }

    fn description(&self) -> &str {
        "Disallow unused variables"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        self.visit(ast, context);
    }
}

impl NoUnusedVariables {
    fn visit(&self, node: &AstNode, context: &mut LintContext) {
        match node {
            AstNode::LetBinding { name, mutable, value, .. } => {
                context.declare_variable(name, *mutable, 0);
                if let Some(val) = value {
                    self.visit(val, context);
                }
            }
            AstNode::Identifier(name) => {
                context.use_variable(name);
            }
            AstNode::Block(stmts) => {
                context.push_scope();
                for stmt in stmts {
                    self.visit(stmt, context);
                }
                context.pop_scope();
            }
            AstNode::FunctionDef { name, params, body, .. } => {
                context.declare_variable(name, false, 0);
                context.push_scope();
                for param in params {
                    context.declare_variable(&param.name, param.is_mut, 0);
                }
                self.visit(body, context);
                context.pop_scope();
            }
            _ => {}
        }
    }
}

// No shadowing
pub struct NoShadowing;

impl LintRule for NoShadowing {
    fn name(&self) -> &str {
        "no-shadowing"
    }

    fn description(&self) -> &str {
        "Disallow variable shadowing"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        self.visit(ast, context);
    }
}

impl NoShadowing {
    fn visit(&self, node: &AstNode, context: &mut LintContext) {
        match node {
            AstNode::LetBinding { name, .. } => {
                if context.is_variable_declared(name) {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: format!("Variable '{}' shadows an existing variable", name),
                        severity: LintSeverity::Warning,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: None,
                    });
                }
            }
            _ => {}
        }
    }
}

// No mutable globals
pub struct NoMutableGlobals;

impl LintRule for NoMutableGlobals {
    fn name(&self) -> &str {
        "no-mutable-globals"
    }

    fn description(&self) -> &str {
        "Disallow mutable global variables"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        match ast {
            AstNode::LetBinding { mutable: true, .. } => {
                if context.function_depth() == 0 {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: "Mutable global variable detected".to_string(),
                        severity: LintSeverity::Warning,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: Some(LintFix {
                            description: "Consider using 'const' instead".to_string(),
                            replacement: String::new(),
                        }),
                    });
                }
            }
            _ => {}
        }
    }
}

// Prefer const
pub struct PreferConst;

impl LintRule for PreferConst {
    fn name(&self) -> &str {
        "prefer-const"
    }

    fn description(&self) -> &str {
        "Prefer const over let for never-reassigned variables"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        // This would require tracking assignments to variables
        // Simplified implementation
    }
}

// No magic numbers
pub struct NoMagicNumbers;

impl LintRule for NoMagicNumbers {
    fn name(&self) -> &str {
        "no-magic-numbers"
    }

    fn description(&self) -> &str {
        "Disallow magic numbers"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        self.visit(ast, context);
    }
}

impl NoMagicNumbers {
    fn visit(&self, node: &AstNode, context: &mut LintContext) {
        match node {
            AstNode::IntegerLiteral(n) => {
                if *n != 0 && *n != 1 && *n != -1 {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: format!("Magic number: {}", n),
                        severity: LintSeverity::Hint,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: Some(LintFix {
                            description: "Consider using a named constant".to_string(),
                            replacement: String::new(),
                        }),
                    });
                }
            }
            _ => {}
        }
    }
}

// Max line length
pub struct MaxLineLength;

impl LintRule for MaxLineLength {
    fn name(&self) -> &str {
        "max-line-length"
    }

    fn description(&self) -> &str {
        "Enforce maximum line length"
    }

    fn check(&self, _ast: &AstNode, _context: &mut LintContext) {
        // Would need source lines to check
    }
}

// Max function length
pub struct MaxFunctionLength;

impl LintRule for MaxFunctionLength {
    fn name(&self) -> &str {
        "max-function-length"
    }

    fn description(&self) -> &str {
        "Enforce maximum function length"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        self.visit(ast, context, 0);
    }
}

impl MaxFunctionLength {
    fn visit(&self, node: &AstNode, context: &mut LintContext, depth: usize) {
        match node {
            AstNode::FunctionDef { name, body, .. } => {
                let stmt_count = self.count_statements(body);
                if stmt_count > context.config().max_function_length {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: format!(
                            "Function '{}' has {} statements (max: {})",
                            name,
                            stmt_count,
                            context.config().max_function_length
                        ),
                        severity: LintSeverity::Warning,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: Some(LintFix {
                            description: "Consider splitting into smaller functions".to_string(),
                            replacement: String::new(),
                        }),
                    });
                }
            }
            _ => {}
        }
    }

    fn count_statements(&self, node: &AstNode) -> usize {
        match node {
            AstNode::Block(stmts) => stmts.iter().map(|s| self.count_statements(s)).sum(),
            AstNode::FunctionDef { body, .. } => 1 + self.count_statements(body),
            _ => 1,
        }
    }
}

// Max complexity
pub struct MaxComplexity;

impl LintRule for MaxComplexity {
    fn name(&self) -> &str {
        "max-complexity"
    }

    fn description(&self) -> &str {
        "Enforce maximum cyclomatic complexity"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        self.visit(ast, context);
    }
}

impl MaxComplexity {
    fn visit(&self, node: &AstNode, context: &mut LintContext) {
        match node {
            AstNode::FunctionDef { name, body, .. } => {
                let complexity = self.calculate_complexity(body);
                if complexity > context.config().max_complexity {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: format!(
                            "Function '{}' has complexity {} (max: {})",
                            name,
                            complexity,
                            context.config().max_complexity
                        ),
                        severity: LintSeverity::Warning,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: Some(LintFix {
                            description: "Consider simplifying the function logic".to_string(),
                            replacement: String::new(),
                        }),
                    });
                }
            }
            _ => {}
        }
    }

    fn calculate_complexity(&self, node: &AstNode) -> usize {
        match node {
            AstNode::If { .. } | AstNode::While { .. } | AstNode::For { .. } => {
                1 + self.children_complexity(node)
            }
            AstNode::Match { arms, .. } => {
                arms.len() + self.children_complexity(node)
            }
            AstNode::BinaryOp { op, left, right } => {
                let base = match op {
                    BinaryOp::And | BinaryOp::Or => 1,
                    _ => 0,
                };
                base + self.calculate_complexity(left) + self.calculate_complexity(right)
            }
            _ => self.children_complexity(node),
        }
    }

    fn children_complexity(&self, node: &AstNode) -> usize {
        match node {
            AstNode::Block(stmts) => stmts.iter().map(|s| self.calculate_complexity(s)).sum(),
            AstNode::If { condition, then_branch, else_branch, .. } => {
                self.calculate_complexity(condition)
                    + self.calculate_complexity(then_branch)
                    + self.calculate_complexity(else_branch.as_ref().unwrap_or(&AstNode::NoneLiteral))
            }
            _ => 0,
        }
    }
}

// Max params
pub struct MaxParams;

impl LintRule for MaxParams {
    fn name(&self) -> &str {
        "max-params"
    }

    fn description(&self) -> &str {
        "Enforce maximum function parameters"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        match ast {
            AstNode::FunctionDef { name, params, .. } => {
                if params.len() > context.config().max_params {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: format!(
                            "Function '{}' has {} parameters (max: {})",
                            name,
                            params.len(),
                            context.config().max_params
                        ),
                        severity: LintSeverity::Warning,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: Some(LintFix {
                            description: "Consider using a struct for parameters".to_string(),
                            replacement: String::new(),
                        }),
                    });
                }
            }
            _ => {}
        }
    }
}

// Naming convention
pub struct NamingConventionRule;

impl LintRule for NamingConventionRule {
    fn name(&self) -> &str {
        "naming-convention"
    }

    fn description(&self) -> &str {
        "Enforce naming conventions"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        self.visit(ast, context);
    }
}

impl NamingConventionRule {
    fn visit(&self, node: &AstNode, context: &mut LintContext) {
        match node {
            AstNode::LetBinding { name, .. } => {
                if !self.is_snake_case(name) && !name.starts_with('_') {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: format!("Variable '{}' should be snake_case", name),
                        severity: LintSeverity::Hint,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: Some(LintFix {
                            description: "Convert to snake_case".to_string(),
                            replacement: self.to_snake_case(name),
                        }),
                    });
                }
            }
            AstNode::FunctionDef { name, .. } => {
                if !self.is_snake_case(name) {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: format!("Function '{}' should be snake_case", name),
                        severity: LintSeverity::Hint,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: Some(LintFix {
                            description: "Convert to snake_case".to_string(),
                            replacement: self.to_snake_case(name),
                        }),
                    });
                }
            }
            AstNode::StructDef { name, .. } => {
                if !self.is_pascal_case(name) {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: format!("Struct '{}' should be PascalCase", name),
                        severity: LintSeverity::Hint,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: Some(LintFix {
                            description: "Convert to PascalCase".to_string(),
                            replacement: self.to_pascal_case(name),
                        }),
                    });
                }
            }
            _ => {}
        }
    }

    fn is_snake_case(&self, s: &str) -> bool {
        s.chars().all(|c| c.is_lowercase() || c == '_' || c.is_numeric())
    }

    fn is_pascal_case(&self, s: &str) -> bool {
        s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            && !s.contains('_')
    }

    fn to_snake_case(&self, s: &str) -> String {
        let mut result = String::new();
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap_or(c));
        }
        result
    }

    fn to_pascal_case(&self, s: &str) -> String {
        s.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    Some(first) => {
                        let mut result: String = first.to_uppercase().collect();
                        result.extend(chars);
                        result
                    }
                    None => String::new(),
                }
            })
            .collect()
    }
}

// No empty blocks
pub struct NoEmptyBlocks;

impl LintRule for NoEmptyBlocks {
    fn name(&self) -> &str {
        "no-empty-blocks"
    }

    fn description(&self) -> &str {
        "Disallow empty code blocks"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        match ast {
            AstNode::Block(stmts) if stmts.is_empty() => {
                context.add_diagnostic(LintDiagnostic {
                    rule: self.name().to_string(),
                    message: "Empty code block detected".to_string(),
                    severity: LintSeverity::Warning,
                    line: 0,
                    column: 0,
                    end_line: 0,
                    end_column: 0,
                    fix: Some(LintFix {
                        description: "Add a comment or remove the empty block".to_string(),
                        replacement: String::new(),
                    }),
                });
            }
            _ => {}
        }
    }
}

// No unreachable code
pub struct NoUnreachableCode;

impl LintRule for NoUnreachableCode {
    fn name(&self) -> &str {
        "no-unreachable-code"
    }

    fn description(&self) -> &str {
        "Disallow unreachable code after return/break/continue"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        self.visit(ast, context);
    }
}

impl NoUnreachableCode {
    fn visit(&self, node: &AstNode, context: &mut LintContext) {
        if let AstNode::Block(stmts) = node {
            for (i, stmt) in stmts.iter().enumerate() {
                if self.is_terminating(stmt) && i + 1 < stmts.len() {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: "Unreachable code detected".to_string(),
                        severity: LintSeverity::Warning,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: None,
                    });
                }
            }
        }
    }

    fn is_terminating(&self, node: &AstNode) -> bool {
        matches!(
            node,
            AstNode::Return { .. } | AstNode::Break { .. } | AstNode::Continue
        )
    }
}

// Prefer early return
pub struct PreferEarlyReturn;

impl LintRule for PreferEarlyReturn {
    fn name(&self) -> &str {
        "prefer-early-return"
    }

    fn description(&self) -> &str {
        "Prefer early returns to reduce nesting"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        // Simplified implementation
    }
}

// No nested if
pub struct NoNestedIf;

impl LintRule for NoNestedIf {
    fn name(&self) -> &str {
        "no-nested-if"
    }

    fn description(&self) -> &str {
        "Avoid deeply nested if statements"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        self.visit(ast, context, 0);
    }
}

impl NoNestedIf {
    fn visit(&self, node: &AstNode, context: &mut LintContext, depth: usize) {
        match node {
            AstNode::If { condition, then_branch, else_branch, .. } => {
                if depth > 2 {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: format!("Deeply nested if (depth: {})", depth),
                        severity: LintSeverity::Warning,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: Some(LintFix {
                            description: "Consider using early returns or guard clauses".to_string(),
                            replacement: String::new(),
                        }),
                    });
                }
                self.visit(then_branch, context, depth + 1);
                if let Some(else_b) = else_branch {
                    self.visit(else_b, context, depth + 1);
                }
            }
            AstNode::Block(stmts) => {
                for stmt in stmts {
                    self.visit(stmt, context, depth);
                }
            }
            _ => {}
        }
    }
}

// Use type annotations
pub struct UseTypeAnnotations;

impl LintRule for UseTypeAnnotations {
    fn name(&self) -> &str {
        "use-type-annotations"
    }

    fn description(&self) -> &str {
        "Require type annotations for function parameters"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        match ast {
            AstNode::FunctionDef { name, params, .. } => {
                for param in params {
                    if param.type_annotation.is_none() {
                        context.add_diagnostic(LintDiagnostic {
                            rule: self.name().to_string(),
                            message: format!(
                                "Parameter '{}' in function '{}' lacks type annotation",
                                param.name, name
                            ),
                            severity: LintSeverity::Info,
                            line: 0,
                            column: 0,
                            end_line: 0,
                            end_column: 0,
                            fix: None,
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

// No deep nesting
pub struct NoDeepNesting;

impl LintRule for NoDeepNesting {
    fn name(&self) -> &str {
        "no-deep-nesting"
    }

    fn description(&self) -> &str {
        "Avoid deeply nested code blocks"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        self.visit(ast, context, 0);
    }
}

impl NoDeepNesting {
    fn visit(&self, node: &AstNode, context: &mut LintContext, depth: usize) {
        if depth > context.config().max_depth {
            context.add_diagnostic(LintDiagnostic {
                rule: self.name().to_string(),
                message: format!("Block nesting too deep (depth: {})", depth),
                severity: LintSeverity::Warning,
                line: 0,
                column: 0,
                end_line: 0,
                end_column: 0,
                fix: Some(LintFix {
                    description: "Consider extracting into a separate function".to_string(),
                    replacement: String::new(),
                }),
            });
        }

        match node {
            AstNode::Block(stmts) => {
                for stmt in stmts {
                    self.visit(stmt, context, depth + 1);
                }
            }
            AstNode::If { then_branch, else_branch, .. } => {
                self.visit(then_branch, context, depth + 1);
                if let Some(else_b) = else_branch {
                    self.visit(else_b, context, depth + 1);
                }
            }
            _ => {}
        }
    }
}

// Consistent return
pub struct ConsistentReturn;

impl LintRule for ConsistentReturn {
    fn name(&self) -> &str {
        "consistent-return"
    }

    fn description(&self) -> &str {
        "Require consistent return statements"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        // Would need to analyze all code paths
    }
}

// No duplicate imports
pub struct NoDuplicateImports;

impl LintRule for NoDuplicateImports {
    fn name(&self) -> &str {
        "no-duplicate-imports"
    }

    fn description(&self) -> &str {
        "Disallow duplicate imports"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        // Would need to track imports
    }
}

// Prefer template strings
pub struct PreferTemplateStrings;

impl LintRule for PreferTemplateStrings {
    fn name(&self) -> &str {
        "prefer-template-strings"
    }

    fn description(&self) -> &str {
        "Prefer template strings over concatenation"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        match ast {
            AstNode::BinaryOp {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                if matches!(left.as_ref(), AstNode::StringLiteral(_))
                    || matches!(right.as_ref(), AstNode::StringLiteral(_))
                {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: "Consider using template strings instead of concatenation".to_string(),
                        severity: LintSeverity::Hint,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: None,
                    });
                }
            }
            _ => {}
        }
    }
}

// No explicit bool comparison
pub struct NoExplicitBoolComparison;

impl LintRule for NoExplicitBoolComparison {
    fn name(&self) -> &str {
        "no-explicit-bool-comparison"
    }

    fn description(&self) -> &str {
        "Disallow explicit boolean comparisons"
    }

    fn check(&self, ast: &AstNode, context: &mut LintContext) {
        match ast {
            AstNode::BinaryOp {
                op: BinaryOp::Eq,
                left,
                right,
            } => {
                if matches!(
                    left.as_ref(),
                    AstNode::BoolLiteral(_)
                ) || matches!(
                    right.as_ref(),
                    AstNode::BoolLiteral(_)
                ) {
                    context.add_diagnostic(LintDiagnostic {
                        rule: self.name().to_string(),
                        message: "Avoid explicit boolean comparison".to_string(),
                        severity: LintSeverity::Hint,
                        line: 0,
                        column: 0,
                        end_line: 0,
                        end_column: 0,
                        fix: Some(LintFix {
                            description: "Use the value directly instead of comparing to true/false".to_string(),
                            replacement: String::new(),
                        }),
                    });
                }
            }
            _ => {}
        }
    }
}
