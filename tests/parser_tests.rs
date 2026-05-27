use omega_lang::lexer::scanner::Scanner;
use omega_lang::parser::parser::Parser;
use omega_lang::ast::*;

fn parse(source: &str) -> AstNode {
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    let mut parser = Parser::new(source);
    parser.parse().unwrap()
}

#[test]
fn test_parse_integer_literal() {
    let ast = parse("42");
    match ast {
        AstNode::Program(stmts) => {
            assert_eq!(stmts.len(), 1);
            assert!(matches!(stmts[0], AstNode::IntegerLiteral(42)));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_float_literal() {
    let ast = parse("3.14");
    match ast {
        AstNode::Program(stmts) => {
            assert_eq!(stmts.len(), 1);
            assert!(matches!(stmts[0], AstNode::FloatLiteral(f) if (f - 3.14).abs() < 0.001));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_bool_literal() {
    let ast = parse("true");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::BoolLiteral(true)));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_string_literal() {
    let ast = parse(r#""hello""#);
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::StringLiteral(ref s) if s == "hello"));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_identifier() {
    let ast = parse("foo");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::Identifier(ref s) if s == "foo"));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_let_binding() {
    let ast = parse("let x = 42");
    match ast {
        AstNode::Program(stmts) => {
            assert_eq!(stmts.len(), 1);
            match &stmts[0] {
                AstNode::LetBinding { name, mutable, type_annotation, value } => {
                    assert_eq!(name, "x");
                    assert!(!mutable);
                    assert!(type_annotation.is_none());
                    assert!(value.is_some());
                }
                _ => panic!("Expected LetBinding"),
            }
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_mutable_binding() {
    let ast = parse("let mut x = 42");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::LetBinding { mutable, .. } => assert!(*mutable),
            _ => panic!("Expected LetBinding"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_const_binding() {
    let ast = parse("const PI = 3.14159");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::ConstBinding { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_binary_add() {
    let ast = parse("1 + 2");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::BinaryOp { op, left, right } => {
                assert!(matches!(op, BinaryOp::Add));
                assert!(matches!(left.as_ref(), AstNode::IntegerLiteral(1)));
                assert!(matches!(right.as_ref(), AstNode::IntegerLiteral(2)));
            }
            _ => panic!("Expected BinaryOp"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_binary_mul() {
    let ast = parse("2 * 3");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::BinaryOp { op, .. } => {
                assert!(matches!(op, BinaryOp::Mul));
            }
            _ => panic!("Expected BinaryOp"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_operator_precedence() {
    let ast = parse("1 + 2 * 3");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::BinaryOp { op, left, right } => {
                assert!(matches!(op, BinaryOp::Add));
                assert!(matches!(left.as_ref(), AstNode::IntegerLiteral(1)));
                match right.as_ref() {
                    AstNode::BinaryOp { op, .. } => {
                        assert!(matches!(op, BinaryOp::Mul));
                    }
                    _ => panic!("Expected BinaryOp for right"),
                }
            }
            _ => panic!("Expected BinaryOp"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_unary_neg() {
    let ast = parse("-42");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::UnaryOp { op, operand } => {
                assert!(matches!(op, UnaryOp::Neg));
                assert!(matches!(operand.as_ref(), AstNode::IntegerLiteral(42)));
            }
            _ => panic!("Expected UnaryOp"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_unary_not() {
    let ast = parse("!true");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::UnaryOp { op, .. } => {
                assert!(matches!(op, UnaryOp::Not));
            }
            _ => panic!("Expected UnaryOp"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_comparison() {
    let ast = parse("x > 0");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::BinaryOp { op, .. } => {
                assert!(matches!(op, BinaryOp::Gt));
            }
            _ => panic!("Expected BinaryOp"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_logical_and() {
    let ast = parse("a && b");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::BinaryOp { op, .. } => {
                assert!(matches!(op, BinaryOp::And));
            }
            _ => panic!("Expected BinaryOp"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_function_def() {
    let ast = parse("fn add(a: i64, b: i64) -> i64 { return a + b }");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::FunctionDef { name, params, return_type, body, is_async, is_pub } => {
                assert_eq!(name, "add");
                assert_eq!(params.len(), 2);
                assert!(return_type.is_some());
                assert!(!is_async);
                assert!(!is_pub);
            }
            _ => panic!("Expected FunctionDef"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_public_function() {
    let ast = parse("pub fn greet() { println(\"hello\") }");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::FunctionDef { is_pub, .. } => {
                assert!(*is_pub);
            }
            _ => panic!("Expected FunctionDef"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_async_function() {
    let ast = parse("async fn fetch() { await http.get(url) }");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::FunctionDef { is_async, .. } => {
                assert!(*is_async);
            }
            _ => panic!("Expected FunctionDef"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_if_expression() {
    let ast = parse("if x > 0 { println(x) }");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::If { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_if_else() {
    let ast = parse("if x > 0 { x } else { -x }");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::If { else_branch, .. } => {
                assert!(else_branch.is_some());
            }
            _ => panic!("Expected If"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_while_loop() {
    let ast = parse("while x > 0 { x -= 1 }");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::While { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_for_loop() {
    let ast = parse("for i in 0..10 { println(i) }");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::For { variable, .. } => {
                assert_eq!(variable, "i");
            }
            _ => panic!("Expected For"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_return() {
    let ast = parse("return 42");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::Return { value } => {
                assert!(value.is_some());
            }
            _ => panic!("Expected Return"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_array_literal() {
    let ast = parse("[1, 2, 3]");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::Array(elements) => {
                assert_eq!(elements.len(), 3);
            }
            _ => panic!("Expected Array"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_empty_array() {
    let ast = parse("[]");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::Array(elements) => {
                assert_eq!(elements.len(), 0);
            }
            _ => panic!("Expected Array"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_map_literal() {
    let ast = parse(r#"{"a": 1, "b": 2}"#);
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::Map(entries) => {
                assert_eq!(entries.len(), 2);
            }
            _ => panic!("Expected Map"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_tuple_literal() {
    let ast = parse("(1, 2, 3)");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::Tuple(elements) => {
                assert_eq!(elements.len(), 3);
            }
            _ => panic!("Expected Tuple"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_function_call() {
    let ast = parse("add(1, 2)");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::Call { function, args, .. } => {
                assert!(matches!(function.as_ref(), AstNode::Identifier(ref s) if s == "add"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_struct_definition() {
    let ast = parse("struct Point { x: f64, y: f64 }");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::StructDef { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_enum_definition() {
    let ast = parse("enum Option<T> { Some(T), None }");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::EnumDef { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_trait_definition() {
    let ast = parse("trait Drawable { fn draw(&self) }");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::TraitDef { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_impl_block() {
    let ast = parse("impl Point { fn new(x: f64, y: f64) -> Point { Point { x, y } } }");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::ImplBlock { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_import() {
    let ast = parse("import math");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::Import { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_module() {
    let ast = parse("mod geometry { pub struct Circle { radius: f64 } }");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::Module { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_lambda() {
    let ast = parse("let f = |x| x * 2");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::LetBinding { value, .. } => {
                assert!(matches!(value.as_ref().unwrap().as_ref(), AstNode::Closure { .. }));
            }
            _ => panic!("Expected LetBinding"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_print() {
    let ast = parse("println(\"hello\")");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::Print { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_assign() {
    let ast = parse("x = 42");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::Assign { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_compound_assignment() {
    let ast = parse("x += 1");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::Assign { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_multiple_statements() {
    let ast = parse("let x = 1\nlet y = 2\nlet z = x + y");
    match ast {
        AstNode::Program(stmts) => {
            assert_eq!(stmts.len(), 3);
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_nested_blocks() {
    let ast = parse("{{ let x = 1 }}");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::Block(_)));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_complex_expression() {
    let ast = parse("(a + b) * (c - d) / e");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::BinaryOp { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_string_interpolation() {
    let ast = parse(r#"println("Hello, {}!", name)"#);
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::Print { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_match_expression() {
    let ast = parse("match x { 0 => println(\"zero\"), _ => println(\"other\") }");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::Match { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_try_catch() {
    let ast = parse("try { risky() } catch (e) { handle(e) }");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::Try { .. }));
        }
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_type_annotation() {
    let ast = parse("let x: i64 = 42");
    match ast {
        AstNode::Program(stmts) => match &stmts[0] {
            AstNode::LetBinding { type_annotation, .. } => {
                assert!(type_annotation.is_some());
            }
            _ => panic!("Expected LetBinding"),
        },
        _ => panic!("Expected Program"),
    }
}

#[test]
fn test_parse_generic_function() {
    let ast = parse("fn identity<T>(x: T) -> T { return x }");
    match ast {
        AstNode::Program(stmts) => {
            assert!(matches!(stmts[0], AstNode::FunctionDef { .. }));
        }
        _ => panic!("Expected Program"),
    }
}
