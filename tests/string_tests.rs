use omega_lang::stdlib::string::ops::OmegaString;

#[test]
fn test_string_new() {
    let s = OmegaString::new("hello");
    assert_eq!(s.as_str(), "hello");
}

#[test]
fn test_string_len() {
    let s = OmegaString::new("hello");
    assert_eq!(s.len(), 5);
}

#[test]
fn test_string_is_empty() {
    assert!(OmegaString::new("").is_empty());
    assert!(!OmegaString::new("hello").is_empty());
}

#[test]
fn test_string_to_upper() {
    let s = OmegaString::new("hello");
    assert_eq!(s.to_upper(), "HELLO");
}

#[test]
fn test_string_to_lower() {
    let s = OmegaString::new("HELLO");
    assert_eq!(s.to_lower(), "hello");
}

#[test]
fn test_string_capitalize() {
    let s = OmegaString::new("hello");
    assert_eq!(s.capitalize(), "Hello");
}

#[test]
fn test_string_trim() {
    let s = OmegaString::new("  hello  ");
    assert_eq!(s.trim(), "hello");
}

#[test]
fn test_string_trim_start() {
    let s = OmegaString::new("  hello");
    assert_eq!(s.trim_start(), "hello");
}

#[test]
fn test_string_trim_end() {
    let s = OmegaString::new("hello  ");
    assert_eq!(s.trim_end(), "hello");
}

#[test]
fn test_string_contains() {
    let s = OmegaString::new("hello world");
    assert!(s.contains("world"));
    assert!(!s.contains("xyz"));
}

#[test]
fn test_string_starts_with() {
    let s = OmegaString::new("hello world");
    assert!(s.starts_with("hello"));
    assert!(!s.starts_with("world"));
}

#[test]
fn test_string_ends_with() {
    let s = OmegaString::new("hello world");
    assert!(s.ends_with("world"));
    assert!(!s.ends_with("hello"));
}

#[test]
fn test_string_find() {
    let s = OmegaString::new("hello world");
    assert_eq!(s.find("world"), Some(6));
    assert_eq!(s.find("xyz"), None);
}

#[test]
fn test_string_replace() {
    let s = OmegaString::new("hello world");
    assert_eq!(s.replace("world", "rust"), "hello rust");
}

#[test]
fn test_string_replace_all() {
    let s = OmegaString::new("hello hello hello");
    assert_eq!(s.replace_all("hello", "hi"), "hi hi hi");
}

#[test]
fn test_string_split() {
    let s = OmegaString::new("a,b,c");
    let parts: Vec<&str> = s.split(',').collect();
    assert_eq!(parts, vec!["a", "b", "c"]);
}

#[test]
fn test_string_reverse() {
    let s = OmegaString::new("hello");
    assert_eq!(s.reverse(), "olleh");
}

#[test]
fn test_string_repeat() {
    let s = OmegaString::new("ha");
    assert_eq!(s.repeat(3), "hahaha");
}

#[test]
fn test_string_pad_left() {
    let s = OmegaString::new("hello");
    assert_eq!(s.pad_left(10, ' '), "     hello");
}

#[test]
fn test_string_pad_right() {
    let s = OmegaString::new("hello");
    assert_eq!(s.pad_right(10, ' '), "hello     ");
}

#[test]
fn test_string_substring() {
    let s = OmegaString::new("hello world");
    assert_eq!(s.substring(0, 5), "hello");
}

#[test]
fn test_string_char_at() {
    let s = OmegaString::new("hello");
    assert_eq!(s.char_at(0), Some('h'));
    assert_eq!(s.char_at(4), Some('o'));
    assert_eq!(s.char_at(5), None);
}

#[test]
fn test_string_chars() {
    let s = OmegaString::new("hello");
    let chars: Vec<char> = s.chars().collect();
    assert_eq!(chars, vec!['h', 'e', 'l', 'l', 'o']);
}

#[test]
fn test_string_is_numeric() {
    assert!(OmegaString::new("123").is_numeric());
    assert!(!OmegaString::new("12.3").is_numeric());
    assert!(!OmegaString::new("abc").is_numeric());
}

#[test]
fn test_string_is_alphabetic() {
    assert!(OmegaString::new("abc").is_alphabetic());
    assert!(!OmegaString::new("abc123").is_alphabetic());
}

#[test]
fn test_string_is_alphanumeric() {
    assert!(OmegaString::new("abc123").is_alphanumeric());
    assert!(!OmegaString::new("abc 123").is_alphanumeric());
}

#[test]
fn test_string_word_count() {
    let s = OmegaString::new("hello world foo bar");
    assert_eq!(s.word_count(), 4);
}

#[test]
fn test_string_line_count() {
    let s = OmegaString::new("line1\nline2\nline3");
    assert_eq!(s.line_count(), 3);
}

#[test]
fn test_string_camel_case() {
    let s = OmegaString::new("hello_world");
    assert_eq!(s.to_camel_case(), "helloWorld");
}

#[test]
fn test_string_snake_case() {
    let s = OmegaString::new("helloWorld");
    assert_eq!(s.to_snake_case(), "hello_world");
}

#[test]
fn test_string_kebab_case() {
    let s = OmegaString::new("helloWorld");
    assert_eq!(s.to_kebab_case(), "hello-world");
}

#[test]
fn test_string_title_case() {
    let s = OmegaString::new("hello world");
    assert_eq!(s.to_title_case(), "Hello World");
}

#[test]
fn test_string_wrap() {
    let s = OmegaString::new("hello world foo bar");
    let wrapped = s.wrap(10);
    assert!(wrapped.contains('\n'));
}

#[test]
fn test_string_indent() {
    let s = OmegaString::new("hello\nworld");
    let indented = s.indent(4);
    assert!(indented.starts_with("    hello"));
}

#[test]
fn test_string_dedent() {
    let s = OmegaString::new("    hello\n    world");
    let dedented = s.dedent();
    assert_eq!(dedented, "hello\nworld");
}

#[test]
fn test_string_escape() {
    let s = OmegaString::new("hello\nworld");
    let escaped = s.escape();
    assert!(escaped.contains("\\n"));
}

#[test]
fn test_string_unescape() {
    let s = OmegaString::new("hello\\nworld");
    let unescaped = s.unescape();
    assert!(unescaped.contains('\n'));
}
