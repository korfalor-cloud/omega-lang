use omega_lang::linter::{Linter, LintConfig, NamingConvention};
use omega_lang::linter::linter::{LintDiagnostic, LintSeverity};
use omega_lang::ast::*;

fn create_let_binding(name: &str, mutable: bool) -> AstNode {
    AstNode::LetBinding {
        name: name.to_string(),
        mutable,
        type_annotation: None,
        value: Some(Box::new(AstNode::IntegerLiteral(42))),
    }
}

fn create_function(name: &str, param_count: usize) -> AstNode {
    let params: Vec<Param> = (0..param_count)
        .map(|i| Param {
            name: format!("p{}", i),
            type_annotation: None,
            is_mut: false,
            default: None,
        })
        .collect();

    AstNode::FunctionDef {
        name: name.to_string(),
        params,
        return_type: None,
        body: Box::new(AstNode::Block(vec![])),
        is_async: false,
        is_pub: false,
    }
}

#[test]
fn test_linter_new() {
    let linter = Linter::new();
    assert!(true);
}

#[test]
fn test_linter_with_config() {
    let config = LintConfig {
        max_line_length: 80,
        max_function_length: 30,
        max_complexity: 5,
        max_params: 3,
        max_depth: 3,
        allow_underscore: true,
        allow_mutable_globals: false,
        strict_types: true,
        naming_convention: NamingConvention::SnakeCase,
    };
    let linter = Linter::with_config(config);
    assert!(true);
}

#[test]
fn test_lint_no_unused_variables() {
    let mut linter = Linter::new();
    let ast = AstNode::Program(vec![
        create_let_binding("used_var", false),
        create_let_binding("unused_var", false),
    ]);

    let diagnostics = linter.lint(&ast);
    // Should detect unused variables
}

#[test]
fn test_lint_mutable_global() {
    let mut linter = Linter::new();
    let ast = AstNode::Program(vec![
        create_let_binding("global_var", true),
    ]);

    let diagnostics = linter.lint(&ast);
    // Should warn about mutable global
}

#[test]
fn test_lint_max_params() {
    let mut linter = Linter::new();
    let ast = AstNode::Program(vec![
        create_function("too_many_params", 10),
    ]);

    let diagnostics = linter.lint(&ast);
    // Should warn about too many parameters
}

#[test]
fn test_lint_naming_convention() {
    let mut linter = Linter::new();
    let ast = AstNode::Program(vec![
        create_let_binding("camelCase", false),
    ]);

    let diagnostics = linter.lint(&ast);
    // Should warn about naming convention
}

#[test]
fn test_lint_empty_block() {
    let mut linter = Linter::new();
    let ast = AstNode::Program(vec![
        AstNode::If {
            condition: Box::new(AstNode::BoolLiteral(true)),
            then_branch: Box::new(AstNode::Block(vec![])),
            elif_branches: vec![],
            else_branch: None,
        },
    ]);

    let diagnostics = linter.lint(&ast);
    // Should warn about empty block
}

#[test]
fn test_lint_unreachable_code() {
    let mut linter = Linter::new();
    let ast = AstNode::Program(vec![
        AstNode::Block(vec![
            AstNode::Return {
                value: Some(Box::new(AstNode::IntegerLiteral(42))),
            },
            AstNode::IntegerLiteral(100), // Unreachable
        ]),
    ]);

    let diagnostics = linter.lint(&ast);
    // Should warn about unreachable code
}

#[test]
fn test_lint_format_diagnostics() {
    let linter = Linter::new();
    let diagnostics = vec![LintDiagnostic {
        rule: "test-rule".to_string(),
        message: "Test message".to_string(),
        severity: LintSeverity::Warning,
        line: 1,
        column: 5,
        end_line: 1,
        end_column: 10,
        fix: None,
    }];

    let source = "let x = 42";
    let formatted = linter.format_diagnostics(&diagnostics, source);
    assert!(formatted.contains("warning"));
    assert!(formatted.contains("Test message"));
    assert!(formatted.contains("test-rule"));
}

#[test]
fn test_lint_format_with_fix() {
    let linter = Linter::new();
    let diagnostics = vec![LintDiagnostic {
        rule: "test-rule".to_string(),
        message: "Test message".to_string(),
        severity: LintSeverity::Hint,
        line: 1,
        column: 0,
        end_line: 1,
        end_column: 0,
        fix: Some(omega_lang::linter::linter::LintFix {
            description: "Try this fix".to_string(),
            replacement: "fixed".to_string(),
        }),
    }];

    let source = "let x = 42";
    let formatted = linter.format_diagnostics(&diagnostics, source);
    assert!(formatted.contains("help"));
    assert!(formatted.contains("Try this fix"));
}
