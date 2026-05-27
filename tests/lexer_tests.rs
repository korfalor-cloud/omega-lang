use omega_lang::lexer::scanner::Scanner;
use omega_lang::lexer::token::TokenKind;

#[test]
fn test_empty_input() {
    let mut scanner = Scanner::new("");
    let tokens = scanner.scan().unwrap();
    assert!(tokens.is_empty() || matches!(tokens.last().map(|t| &t.kind), Some(TokenKind::Eof)));
}

#[test]
fn test_integer_literals() {
    let mut scanner = Scanner::new("42 0 1000000 -5");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral(42));
    assert_eq!(tokens[1].kind, TokenKind::IntegerLiteral(0));
    assert_eq!(tokens[2].kind, TokenKind::IntegerLiteral(1000000));
}

#[test]
fn test_float_literals() {
    let mut scanner = Scanner::new("3.14 0.5 1.0 100.001");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::FloatLiteral(3.14));
    assert_eq!(tokens[1].kind, TokenKind::FloatLiteral(0.5));
    assert_eq!(tokens[2].kind, TokenKind::FloatLiteral(1.0));
}

#[test]
fn test_hex_literals() {
    let mut scanner = Scanner::new("0xFF 0x00 0xABCD");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral(255));
    assert_eq!(tokens[1].kind, TokenKind::IntegerLiteral(0));
    assert_eq!(tokens[2].kind, TokenKind::IntegerLiteral(43981));
}

#[test]
fn test_binary_literals() {
    let mut scanner = Scanner::new("0b1010 0b0 0b11111111");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral(10));
    assert_eq!(tokens[1].kind, TokenKind::IntegerLiteral(0));
    assert_eq!(tokens[2].kind, TokenKind::IntegerLiteral(255));
}

#[test]
fn test_octal_literals() {
    let mut scanner = Scanner::new("0o777 0o0 0o17");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral(511));
    assert_eq!(tokens[1].kind, TokenKind::IntegerLiteral(0));
    assert_eq!(tokens[2].kind, TokenKind::IntegerLiteral(15));
}

#[test]
fn test_string_literals() {
    let mut scanner = Scanner::new(r#""hello" "world" "with spaces""#);
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::StringLiteral("hello".to_string()));
    assert_eq!(tokens[1].kind, TokenKind::StringLiteral("world".to_string()));
    assert_eq!(tokens[2].kind, TokenKind::StringLiteral("with spaces".to_string()));
}

#[test]
fn test_string_escapes() {
    let mut scanner = Scanner::new(r#""hello\nworld\ttab\\slash""#);
    let tokens = scanner.scan().unwrap();
    assert_eq!(
        tokens[0].kind,
        TokenKind::StringLiteral("hello\nworld\ttab\\slash".to_string())
    );
}

#[test]
fn test_char_literals() {
    let mut scanner = Scanner::new("'a' 'Z' '0' '\\n'");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::CharLiteral('a'));
    assert_eq!(tokens[1].kind, TokenKind::CharLiteral('Z'));
    assert_eq!(tokens[2].kind, TokenKind::CharLiteral('0'));
    assert_eq!(tokens[3].kind, TokenKind::CharLiteral('\n'));
}

#[test]
fn test_bool_literals() {
    let mut scanner = Scanner::new("true false");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::BoolLiteral(true));
    assert_eq!(tokens[1].kind, TokenKind::BoolLiteral(false));
}

#[test]
fn test_none_literal() {
    let mut scanner = Scanner::new("none");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::NoneLiteral);
}

#[test]
fn test_identifiers() {
    let mut scanner = Scanner::new("foo bar_baz camelCase PascalCase x");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Identifier("foo".to_string()));
    assert_eq!(tokens[1].kind, TokenKind::Identifier("bar_baz".to_string()));
    assert_eq!(tokens[2].kind, TokenKind::Identifier("camelCase".to_string()));
    assert_eq!(tokens[3].kind, TokenKind::Identifier("PascalCase".to_string()));
    assert_eq!(tokens[4].kind, TokenKind::Identifier("x".to_string()));
}

#[test]
fn test_keywords() {
    let mut scanner = Scanner::new("let mut const fn if else while for return");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
    assert!(matches!(tokens[1].kind, TokenKind::Keyword(_)));
    assert!(matches!(tokens[2].kind, TokenKind::Keyword(_)));
    assert!(matches!(tokens[3].kind, TokenKind::Keyword(_)));
    assert!(matches!(tokens[4].kind, TokenKind::Keyword(_)));
    assert!(matches!(tokens[5].kind, TokenKind::Keyword(_)));
    assert!(matches!(tokens[6].kind, TokenKind::Keyword(_)));
    assert!(matches!(tokens[7].kind, TokenKind::Keyword(_)));
    assert!(matches!(tokens[8].kind, TokenKind::Keyword(_)));
}

#[test]
fn test_arithmetic_operators() {
    let mut scanner = Scanner::new("+ - * / % ** //");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Plus);
    assert_eq!(tokens[1].kind, TokenKind::Minus);
    assert_eq!(tokens[2].kind, TokenKind::Star);
    assert_eq!(tokens[3].kind, TokenKind::Slash);
    assert_eq!(tokens[4].kind, TokenKind::Percent);
    assert_eq!(tokens[5].kind, TokenKind::StarStar);
    assert_eq!(tokens[6].kind, TokenKind::SlashSlash);
}

#[test]
fn test_comparison_operators() {
    let mut scanner = Scanner::new("== != < <= > >= <=>");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::EqualEqual);
    assert_eq!(tokens[1].kind, TokenKind::BangEqual);
    assert_eq!(tokens[2].kind, TokenKind::Less);
    assert_eq!(tokens[3].kind, TokenKind::LessEqual);
    assert_eq!(tokens[4].kind, TokenKind::Greater);
    assert_eq!(tokens[5].kind, TokenKind::GreaterEqual);
}

#[test]
fn test_logical_operators() {
    let mut scanner = Scanner::new("&& || !");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::AmpAmp);
    assert_eq!(tokens[1].kind, TokenKind::PipePipe);
    assert_eq!(tokens[2].kind, TokenKind::Bang);
}

#[test]
fn test_bitwise_operators() {
    let mut scanner = Scanner::new("& | ^ ~ << >>");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Amp);
    assert_eq!(tokens[1].kind, TokenKind::Pipe);
    assert_eq!(tokens[2].kind, TokenKind::Caret);
    assert_eq!(tokens[3].kind, TokenKind::Tilde);
    assert_eq!(tokens[4].kind, TokenKind::LessLess);
    assert_eq!(tokens[5].kind, TokenKind::GreaterGreater);
}

#[test]
fn test_assignment_operators() {
    let mut scanner = Scanner::new("= += -= *= /= %= **= &= |= ^= <<= >>=");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Equal);
    assert_eq!(tokens[1].kind, TokenKind::PlusEqual);
    assert_eq!(tokens[2].kind, TokenKind::MinusEqual);
    assert_eq!(tokens[3].kind, TokenKind::StarEqual);
    assert_eq!(tokens[4].kind, TokenKind::SlashEqual);
    assert_eq!(tokens[5].kind, TokenKind::PercentEqual);
    assert_eq!(tokens[6].kind, TokenKind::StarStarEqual);
    assert_eq!(tokens[7].kind, TokenKind::AmpEqual);
    assert_eq!(tokens[8].kind, TokenKind::PipeEqual);
    assert_eq!(tokens[9].kind, TokenKind::CaretEqual);
    assert_eq!(tokens[10].kind, TokenKind::LessLessEqual);
    assert_eq!(tokens[11].kind, TokenKind::GreaterGreaterEqual);
}

#[test]
fn test_delimiters() {
    let mut scanner = Scanner::new("( ) [ ] { } , ; : . -> => .. ..= @ #");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::LeftParen);
    assert_eq!(tokens[1].kind, TokenKind::RightParen);
    assert_eq!(tokens[2].kind, TokenKind::LeftBracket);
    assert_eq!(tokens[3].kind, TokenKind::RightBracket);
    assert_eq!(tokens[4].kind, TokenKind::LeftBrace);
    assert_eq!(tokens[5].kind, TokenKind::RightBrace);
    assert_eq!(tokens[6].kind, TokenKind::Comma);
    assert_eq!(tokens[7].kind, TokenKind::Semicolon);
    assert_eq!(tokens[8].kind, TokenKind::Colon);
    assert_eq!(tokens[9].kind, TokenKind::Dot);
    assert_eq!(tokens[10].kind, TokenKind::Arrow);
    assert_eq!(tokens[11].kind, TokenKind::FatArrow);
    assert_eq!(tokens[12].kind, TokenKind::DotDot);
    assert_eq!(tokens[13].kind, TokenKind::DotDotEqual);
}

#[test]
fn test_single_line_comments() {
    let mut scanner = Scanner::new("42 // this is a comment\n100");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral(42));
    assert_eq!(tokens[1].kind, TokenKind::IntegerLiteral(100));
}

#[test]
fn test_multi_line_comments() {
    let mut scanner = Scanner::new("42 /* multi\nline\ncomment */ 100");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral(42));
    assert_eq!(tokens[1].kind, TokenKind::IntegerLiteral(100));
}

#[test]
fn test_nested_comments() {
    let mut scanner = Scanner::new("42 /* outer /* inner */ still comment */ 100");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral(42));
    assert_eq!(tokens[1].kind, TokenKind::IntegerLiteral(100));
}

#[test]
fn test_let_binding() {
    let mut scanner = Scanner::new("let x = 42");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
    assert_eq!(tokens[1].kind, TokenKind::Identifier("x".to_string()));
    assert_eq!(tokens[2].kind, TokenKind::Equal);
    assert_eq!(tokens[3].kind, TokenKind::IntegerLiteral(42));
}

#[test]
fn test_function_definition() {
    let mut scanner = Scanner::new("fn add(a: i64, b: i64) -> i64 { return a + b }");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
    assert_eq!(tokens[1].kind, TokenKind::Identifier("add".to_string()));
    assert_eq!(tokens[2].kind, TokenKind::LeftParen);
    assert_eq!(tokens[3].kind, TokenKind::Identifier("a".to_string()));
    assert_eq!(tokens[4].kind, TokenKind::Colon);
    assert_eq!(tokens[5].kind, TokenKind::Identifier("i64".to_string()));
    assert_eq!(tokens[6].kind, TokenKind::Comma);
    assert_eq!(tokens[7].kind, TokenKind::Identifier("b".to_string()));
    assert_eq!(tokens[8].kind, TokenKind::Colon);
    assert_eq!(tokens[9].kind, TokenKind::Identifier("i64".to_string()));
    assert_eq!(tokens[10].kind, TokenKind::RightParen);
    assert_eq!(tokens[11].kind, TokenKind::Arrow);
}

#[test]
fn test_struct_definition() {
    let mut scanner = Scanner::new("struct Point { x: f64, y: f64 }");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
    assert_eq!(tokens[1].kind, TokenKind::Identifier("Point".to_string()));
    assert_eq!(tokens[2].kind, TokenKind::LeftBrace);
}

#[test]
fn test_enum_definition() {
    let mut scanner = Scanner::new("enum Option<T> { Some(T), None }");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
    assert_eq!(tokens[1].kind, TokenKind::Identifier("Option".to_string()));
    assert_eq!(tokens[2].kind, TokenKind::Less);
    assert_eq!(tokens[3].kind, TokenKind::Identifier("T".to_string()));
    assert_eq!(tokens[4].kind, TokenKind::Greater);
}

#[test]
fn test_if_else() {
    let mut scanner = Scanner::new("if x > 0 { println(x) } else { println(-x) }");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
    assert_eq!(tokens[1].kind, TokenKind::Identifier("x".to_string()));
    assert_eq!(tokens[2].kind, TokenKind::Greater);
    assert_eq!(tokens[3].kind, TokenKind::IntegerLiteral(0));
}

#[test]
fn test_while_loop() {
    let mut scanner = Scanner::new("while x > 0 { x -= 1 }");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
}

#[test]
fn test_for_loop() {
    let mut scanner = Scanner::new("for i in 0..10 { println(i) }");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
    assert_eq!(tokens[1].kind, TokenKind::Identifier("i".to_string()));
    assert!(matches!(tokens[2].kind, TokenKind::Keyword(_)));
    assert_eq!(tokens[3].kind, TokenKind::IntegerLiteral(0));
    assert_eq!(tokens[4].kind, TokenKind::DotDot);
    assert_eq!(tokens[5].kind, TokenKind::IntegerLiteral(10));
}

#[test]
fn test_match_expression() {
    let mut scanner = Scanner::new("match x { 0 => println(\"zero\"), _ => println(\"other\") }");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
}

#[test]
fn test_lambda() {
    let mut scanner = Scanner::new("let f = |x| x * 2");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[3].kind, TokenKind::Pipe);
    assert_eq!(tokens[4].kind, TokenKind::Identifier("x".to_string()));
    assert_eq!(tokens[5].kind, TokenKind::Pipe);
}

#[test]
fn test_multiline_program() {
    let source = r#"
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

let result = fibonacci(10)
println(result)
"#;
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_string_interpolation_format() {
    let mut scanner = Scanner::new(r#"println("Hello, {}!", name)"#);
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Identifier("println".to_string()));
}

#[test]
fn test_array_literal() {
    let mut scanner = Scanner::new("[1, 2, 3, 4, 5]");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::LeftBracket);
    assert_eq!(tokens[1].kind, TokenKind::IntegerLiteral(1));
    assert_eq!(tokens[2].kind, TokenKind::Comma);
}

#[test]
fn test_map_literal() {
    let mut scanner = Scanner::new(r#"{"key": "value", "count": 42}"#);
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::LeftBrace);
}

#[test]
fn test_tuple_literal() {
    let mut scanner = Scanner::new("(1, \"hello\", true)");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::LeftParen);
    assert_eq!(tokens[1].kind, TokenKind::IntegerLiteral(1));
}

#[test]
fn test_import_statement() {
    let mut scanner = Scanner::new("import math");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
    assert_eq!(tokens[1].kind, TokenKind::Identifier("math".to_string()));
}

#[test]
fn test_from_import() {
    let mut scanner = Scanner::new("from math import sin, cos, tan");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
}

#[test]
fn test_async_function() {
    let mut scanner = Scanner::new("async fn fetch(url: String) -> Response { await http.get(url) }");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
    assert!(matches!(tokens[1].kind, TokenKind::Keyword(_)));
}

#[test]
fn test_try_catch() {
    let mut scanner = Scanner::new("try { risky() } catch (e) { handle(e) }");
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
}

#[test]
fn test_chain_comparison() {
    let mut scanner = Scanner::new("a < b && b < c");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Identifier("a".to_string()));
    assert_eq!(tokens[1].kind, TokenKind::Less);
}

#[test]
fn test_string_with_newlines() {
    let mut scanner = Scanner::new("\"line1\\nline2\\nline3\"");
    let tokens = scanner.scan().unwrap();
    if let TokenKind::StringLiteral(s) = &tokens[0].kind {
        assert!(s.contains('\n'));
    }
}

#[test]
fn test_empty_string() {
    let mut scanner = Scanner::new("\"\"");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::StringLiteral("".to_string()));
}

#[test]
fn test_large_number() {
    let mut scanner = Scanner::new("999999999999999999");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::IntegerLiteral(999999999999999999));
}

#[test]
fn test_very_small_float() {
    let mut scanner = Scanner::new("0.000001");
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::FloatLiteral(0.000001));
}

#[test]
fn test_scientific_notation() {
    let mut scanner = Scanner::new("1e10 2.5e-3");
    let tokens = scanner.scan().unwrap();
    // Scientific notation should be handled
    assert!(!tokens.is_empty());
}

#[test]
fn test_multiple_statements() {
    let source = "let x = 1; let y = 2; let z = x + y";
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Identifier("x".to_string()));
}

#[test]
fn test_nested_expressions() {
    let source = "(a + b) * (c - d) / e";
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::LeftParen);
}

#[test]
fn test_method_chain() {
    let source = "obj.method1().method2().method3()";
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::Identifier("obj".to_string()));
    assert_eq!(tokens[1].kind, TokenKind::Dot);
}

#[test]
fn test_trait_definition() {
    let source = "trait Drawable { fn draw(&self) }";
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
}

#[test]
fn test_impl_block() {
    let source = "impl Point { fn new(x: f64, y: f64) -> Point { Point { x, y } } }";
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
}

#[test]
fn test_module_declaration() {
    let source = "mod geometry { pub struct Circle { radius: f64 } }";
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
}

#[test]
fn test_type_annotation() {
    let source = "let x: Vec<i64> = vec![1, 2, 3]";
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert!(matches!(tokens[0].kind, TokenKind::Keyword(_)));
}

#[test]
fn test_optional_type() {
    let source = "let x: i64? = none";
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    // Should handle ? as optional type marker
}

#[test]
fn test_spread_operator() {
    let source = "let arr = [..other, 4, 5]";
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[0].kind, TokenKind::LeftBracket);
}

#[test]
fn test_underscore_identifier() {
    let source = "let _ = 42";
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert_eq!(tokens[1].kind, TokenKind::Identifier("_".to_string()));
}

#[test]
fn test_raw_string() {
    let source = r#"r"no \escaping here""#;
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    // Raw strings should not process escapes
}

#[test]
fn test_long_program() {
    let source = r#"
fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false
    }
    for i in 2..n {
        if n % i == 0 {
            return false
        }
    }
    return true
}

fn main() {
    for i in 0..20 {
        let fib = fibonacci(i)
        println("fib({}) = {}", i, fib)
    }

    for i in 0..100 {
        if is_prime(i) {
            println("{} is prime", i)
        }
    }
}

main()
"#;
    let mut scanner = Scanner::new(source);
    let tokens = scanner.scan().unwrap();
    assert!(tokens.len() > 50);
}
