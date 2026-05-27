use crate::errors::{Position, Span};
use super::token::{Token, TokenKind, Keyword};
use std::collections::VecDeque;

pub struct Scanner {
    source: Vec<char>,
    start: usize,
    current: usize,
    line: usize,
    col: usize,
    indent_stack: Vec<usize>,
    pending_tokens: VecDeque<Token>,
    at_line_start: bool,
    track_indent: bool,
    paren_depth: usize,
    brace_depth: usize,
    bracket_depth: usize,
}

impl Scanner {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.chars().collect(),
            start: 0,
            current: 0,
            line: 1,
            col: 1,
            indent_stack: vec![0],
            pending_tokens: VecDeque::new(),
            at_line_start: true,
            track_indent: true,
            paren_depth: 0,
            brace_depth: 0,
            bracket_depth: 0,
        }
    }

    pub fn scan_all(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            let is_eof = matches!(token.kind, TokenKind::Eof);
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        tokens
    }

    pub fn next_token(&mut self) -> Token {
        if let Some(token) = self.pending_tokens.pop_front() {
            return token;
        }

        self.skip_whitespace_and_comments();

        self.start = self.current;

        if self.is_at_end() {
            return self.make_token(TokenKind::Eof);
        }

        let ch = self.advance();

        match ch {
            '\n' => self.handle_newline(),
            '(' => { self.paren_depth += 1; self.make_token(TokenKind::LeftParen) }
            ')' => { self.paren_depth = self.paren_depth.saturating_sub(1); self.make_token(TokenKind::RightParen) }
            '{' => { self.brace_depth += 1; self.make_token(TokenKind::LeftBrace) }
            '}' => { self.brace_depth = self.brace_depth.saturating_sub(1); self.make_token(TokenKind::RightBrace) }
            '[' => { self.bracket_depth += 1; self.make_token(TokenKind::LeftBracket) }
            ']' => { self.bracket_depth = self.bracket_depth.saturating_sub(1); self.make_token(TokenKind::RightBracket) }
            ';' => self.make_token(TokenKind::Semicolon),
            ',' => self.make_token(TokenKind::Comma),
            '@' => self.make_token(TokenKind::At),
            '#' => self.make_token(TokenKind::Hash),
            '$' => self.make_token(TokenKind::Dollar),
            '\\' => self.make_token(TokenKind::Backslash),
            '~' => self.make_token(TokenKind::Tilde),
            '+' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::PlusEqual)
                } else {
                    self.make_token(TokenKind::Plus)
                }
            }
            '-' => {
                if self.match_char('>') {
                    self.make_token(TokenKind::Arrow)
                } else if self.match_char('=') {
                    self.make_token(TokenKind::MinusEqual)
                } else {
                    self.make_token(TokenKind::Minus)
                }
            }
            '*' => {
                if self.match_char('*') {
                    if self.match_char('=') {
                        self.make_token(TokenKind::StarStarEqual)
                    } else {
                        self.make_token(TokenKind::StarStar)
                    }
                } else if self.match_char('=') {
                    self.make_token(TokenKind::StarEqual)
                } else {
                    self.make_token(TokenKind::Star)
                }
            }
            '/' => {
                if self.match_char('/') {
                    if self.match_char('=') {
                        self.make_token(TokenKind::SlashSlashEqual)
                    } else {
                        self.make_token(TokenKind::SlashSlash)
                    }
                } else if self.match_char('=') {
                    self.make_token(TokenKind::SlashEqual)
                } else {
                    self.make_token(TokenKind::Slash)
                }
            }
            '%' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::PercentEqual)
                } else {
                    self.make_token(TokenKind::Percent)
                }
            }
            '&' => {
                if self.match_char('&') {
                    self.make_token(TokenKind::AmpAmp)
                } else if self.match_char('=') {
                    self.make_token(TokenKind::AmpersandEqual)
                } else {
                    self.make_token(TokenKind::Ampersand)
                }
            }
            '|' => {
                if self.match_char('|') {
                    self.make_token(TokenKind::PipePipe)
                } else if self.match_char('=') {
                    self.make_token(TokenKind::PipeEqual)
                } else {
                    self.make_token(TokenKind::Pipe)
                }
            }
            '^' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::CaretEqual)
                } else {
                    self.make_token(TokenKind::Caret)
                }
            }
            '=' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::EqualEqual)
                } else if self.match_char('>') {
                    self.make_token(TokenKind::FatArrow)
                } else {
                    self.make_token(TokenKind::Equal)
                }
            }
            '!' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::BangEqual)
                } else {
                    self.make_token(TokenKind::Bang)
                }
            }
            '<' => {
                if self.match_char('=') {
                    if self.match_char('>') {
                        self.make_token(TokenKind::Spaceship)
                    } else {
                        self.make_token(TokenKind::LessEqual)
                    }
                } else if self.match_char('<') {
                    if self.match_char('=') {
                        self.make_token(TokenKind::LessLessEqual)
                    } else {
                        self.make_token(TokenKind::LessLess)
                    }
                } else {
                    self.make_token(TokenKind::Less)
                }
            }
            '>' => {
                if self.match_char('=') {
                    self.make_token(TokenKind::GreaterEqual)
                } else if self.match_char('>') {
                    if self.match_char('=') {
                        self.make_token(TokenKind::GreaterGreaterEqual)
                    } else {
                        self.make_token(TokenKind::GreaterGreater)
                    }
                } else {
                    self.make_token(TokenKind::Greater)
                }
            }
            '?' => {
                if self.match_char('.') {
                    self.make_token(TokenKind::QuestionDot)
                } else if self.match_char('?') {
                    self.make_token(TokenKind::QuestionQuestion)
                } else {
                    self.make_token(TokenKind::Question)
                }
            }
            '.' => {
                if self.match_char('.') {
                    if self.match_char('.') {
                        self.make_token(TokenKind::DotDotDot)
                    } else if self.match_char('=') {
                        self.make_token(TokenKind::DotDotEqual)
                    } else {
                        self.make_token(TokenKind::DotDot)
                    }
                } else {
                    self.make_token(TokenKind::Dot)
                }
            }
            ':' => {
                if self.match_char(':') {
                    // :: handled separately if needed
                    self.make_token(TokenKind::Colon) // simplified
                } else {
                    self.make_token(TokenKind::Colon)
                }
            }
            '"' => self.scan_string(),
            '\'' => self.scan_char_or_lifetime(),
            '`' => self.scan_raw_string(),
            '0'..='9' => self.scan_number(),
            'a'..='z' | 'A'..='Z' | '_' | '$' => self.scan_identifier(),
            _ => Token::new(
                TokenKind::Error(format!("Unexpected character '{}'", ch)),
                self.source[self.start..self.current].iter().collect(),
                self.line,
                self.col - 1,
                self.start,
            ),
        }
    }

    fn handle_newline(&mut self) -> Token {
        if self.in_grouping() {
            self.skip_whitespace_and_comments();
            return self.next_token();
        }

        let indent = self.measure_indent();
        let current_indent = *self.indent_stack.last().unwrap_or(&0);

        if indent > current_indent {
            self.indent_stack.push(indent);
            self.pending_tokens.push_back(Token::new(
                TokenKind::Indent,
                String::new(),
                self.line,
                1,
                self.current,
            ));
        } else {
            while indent < *self.indent_stack.last().unwrap_or(&0) {
                self.indent_stack.pop();
                self.pending_tokens.push_back(Token::new(
                    TokenKind::Dedent,
                    String::new(),
                    self.line,
                    1,
                    self.current,
                ));
            }
        }

        self.pending_tokens.push_back(Token::new(
            TokenKind::Newline,
            String::new(),
            self.line,
            1,
            self.current,
        ));

        self.pending_tokens.pop_front().unwrap_or_else(|| self.make_token(TokenKind::Newline))
    }

    fn measure_indent(&mut self) -> usize {
        let mut indent = 0;
        while !self.is_at_end() {
            match self.peek() {
                ' ' => { indent += 1; self.advance(); }
                '\t' => { indent += 4; self.advance(); }
                _ => break,
            }
        }
        indent
    }

    fn in_grouping(&self) -> bool {
        self.paren_depth > 0 || self.brace_depth > 0 || self.bracket_depth > 0
    }

    fn scan_string(&mut self) -> Token {
        let mut value = String::new();
        let mut is_multiline = false;

        if self.peek() == '"' && self.peek_next() == '"' {
            is_multiline = true;
            self.advance();
            self.advance();
            self.skip_whitespace();
            if self.peek() == '\n' {
                self.advance();
            }
        }

        loop {
            if self.is_at_end() {
                return self.error_token("Unterminated string");
            }

        let ch = self.advance();
        match ch {
            '"' => {
                if is_multiline {
                    if self.peek() == '"' && self.peek_next() == '"' {
                        self.advance();
                        self.advance();
                        break;
                    }
                } else {
                    break;
                }
                value.push(ch);
            }
            '\\' => {
                let escaped = self.advance();
                match escaped {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    '\'' => value.push('\''),
                    '0' => value.push('\0'),
                    'a' => value.push('\x07'),
                    'b' => value.push('\x08'),
                    'f' => value.push('\x0C'),
                    'v' => value.push('\x0B'),
                    'e' => value.push('\x1B'),
                    'x' => {
                        let hex = self.scan_hex_escape(2);
                        value.push(hex);
                    }
                    'u' => {
                        let code = self.scan_unicode_escape();
                        value.push(code);
                    }
                    '\n' => {
                        if is_multiline {
                            value.push('\n');
                        }
                    }
                    _ => {
                        return self.error_token(&format!("Invalid escape sequence '\\{}'", escaped));
                    }
                }
            }
            '\n' => {
                if !is_multiline {
                    return self.error_token("Unterminated string (use triple quotes for multiline)");
                }
                value.push(ch);
            }
            _ => value.push(ch),
        }
    }

        self.make_token(TokenKind::String(value))
    }

    fn scan_hex_escape(&mut self, count: usize) -> char {
        let mut hex = String::new();
        for _ in 0..count {
            if self.is_at_end() || !self.peek().is_ascii_hexdigit() {
                return '\0';
            }
            hex.push(self.advance());
        }
        u8::from_str_radix(&hex, 16).unwrap_or(0) as char
    }

    fn scan_unicode_escape(&mut self) -> char {
        if !self.match_char('{') {
            return '\0';
        }
        let mut hex = String::new();
        while !self.is_at_end() && self.peek() != '}' {
            if self.peek().is_ascii_hexdigit() {
                hex.push(self.advance());
            } else {
                return '\0';
            }
        }
        self.match_char('}');
        u32::from_str_radix(&hex, 16)
            .ok()
            .and_then(|cp| char::from_u32(cp))
            .unwrap_or('\0')
    }

    fn scan_char_or_lifetime(&mut self) -> Token {
        let start = self.current;

        if self.peek().is_alphanumeric() && self.peek_next() == '\'' {
            let ch = self.advance();
            self.advance(); // consume closing '
            return self.make_token(TokenKind::Char(ch));
        }

        if self.peek().is_alphabetic() || self.peek() == '_' {
            while !self.is_at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
                self.advance();
            }
            let lifetime: String = self.source[start..self.current].iter().collect();
            return self.make_token(TokenKind::Identifier(format!("'{}", lifetime)));
        }

        self.error_token("Invalid character literal or lifetime")
    }

    fn scan_raw_string(&mut self) -> Token {
        let mut value = String::new();
        loop {
            if self.is_at_end() {
                return self.error_token("Unterminated raw string");
            }
            let ch = self.advance();
            if ch == '`' {
                break;
            }
            value.push(ch);
        }
        self.make_token(TokenKind::String(value))
    }

    fn scan_number(&mut self) -> Token {
        let mut is_float = false;
        let mut base = 10;

        if self.previous() == '0' {
            match self.peek() {
                'x' | 'X' => {
                    base = 16;
                    self.advance();
                    self.scan_digits(16);
                }
                'o' | 'O' => {
                    base = 8;
                    self.advance();
                    self.scan_digits(8);
                }
                'b' | 'B' => {
                    base = 2;
                    self.advance();
                    self.scan_digits(2);
                }
                _ => {}
            }
        } else {
            self.scan_digits(10);
        }

        if base == 10 && self.peek() == '.' && self.peek_next() != '.' {
            is_float = true;
            self.advance();
            self.scan_digits(10);

            if self.peek() == 'e' || self.peek() == 'E' {
                self.advance();
                if self.peek() == '+' || self.peek() == '-' {
                    self.advance();
                }
                self.scan_digits(10);
            }
        }

        if self.peek() == 'i' || self.peek() == 'u' || self.peek() == 'f' {
            let suffix_start = self.current;
            while !self.is_at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
                self.advance();
            }
            let suffix: String = self.source[suffix_start..self.current].iter().collect();
            match suffix.as_str() {
                "f32" | "f64" => is_float = true,
                "i8" | "i16" | "i32" | "i64" | "isize" |
                "u8" | "u16" | "u32" | "u64" | "usize" => {}
                _ => {
                    return self.error_token(&format!("Unknown numeric suffix '{}'", suffix));
                }
            }
        }

        let lexeme: String = self.source[self.start..self.current].iter().collect();

        if is_float {
            match lexeme.parse::<f64>() {
                Ok(value) => self.make_token(TokenKind::Float(value)),
                Err(_) => self.error_token("Invalid float literal"),
            }
        } else {
            let clean: String = lexeme.chars().filter(|c| *c != '_').collect();
            match i64::from_str_radix(&clean.trim_start_matches("0x").trim_start_matches("0o").trim_start_matches("0b"), base) {
                Ok(value) => self.make_token(TokenKind::Integer(value)),
                Err(_) => {
                    if clean.len() > 20 {
                        self.make_token(TokenKind::BigInt(clean))
                    } else {
                        self.error_token("Invalid integer literal")
                    }
                }
            }
        }
    }

    fn scan_digits(&mut self, base: u32) {
        loop {
            if self.is_at_end() {
                break;
            }
            let ch = self.peek();
            if ch == '_' {
                self.advance();
                continue;
            }
            match base {
                2 => if ch != '0' && ch != '1' { break; },
                8 => if !('0'..='7').contains(&ch) { break; },
                10 => if !ch.is_ascii_digit() { break; },
                16 => if !ch.is_ascii_hexdigit() { break; },
                _ => break,
            }
            self.advance();
        }
    }

    fn scan_identifier(&mut self) -> Token {
        while !self.is_at_end() && (self.peek().is_alphanumeric() || self.peek() == '_' || self.peek() == '$') {
            self.advance();
        }

        let text: String = self.source[self.start..self.current].iter().collect();

        match text.as_str() {
            "true" => self.make_token(TokenKind::Bool(true)),
            "false" => self.make_token(TokenKind::Bool(false)),
            "none" => self.make_token(TokenKind::None),
            "and" => self.make_token(TokenKind::AmpAmp),
            "or" => self.make_token(TokenKind::PipePipe),
            "not" => self.make_token(TokenKind::Bang),
            _ => {
                if let Some(keyword) = Keyword::from_str(&text) {
                    self.make_token(TokenKind::Keyword(keyword))
                } else {
                    self.make_token(TokenKind::Identifier(text))
                }
            }
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            if self.is_at_end() {
                return;
            }

            match self.peek() {
                ' ' | '\r' | '\t' => {
                    self.advance();
                }
                '/' if self.peek_next() == '/' => {
                    while !self.is_at_end() && self.peek() != '\n' {
                        self.advance();
                    }
                }
                '/' if self.peek_next() == '*' => {
                    self.advance();
                    self.advance();
                    self.skip_block_comment();
                }
                '#' => {
                    while !self.is_at_end() && self.peek() != '\n' {
                        self.advance();
                    }
                }
                _ => return,
            }
        }
    }

    fn skip_block_comment(&mut self) {
        let mut depth = 1;
        while !self.is_at_end() && depth > 0 {
            if self.peek() == '/' && self.peek_next() == '*' {
                self.advance();
                self.advance();
                depth += 1;
            } else if self.peek() == '*' && self.peek_next() == '/' {
                self.advance();
                self.advance();
                depth -= 1;
            } else {
                self.advance();
            }
        }
    }

    fn advance(&mut self) -> char {
        let ch = self.source[self.current];
        self.current += 1;
        self.col += 1;
        ch
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() || self.source[self.current] != expected {
            return false;
        }
        self.current += 1;
        self.col += 1;
        true
    }

    fn peek(&self) -> char {
        if self.is_at_end() { '\0' } else { self.source[self.current] }
    }

    fn peek_next(&self) -> char {
        if self.current + 1 >= self.source.len() { '\0' } else { self.source[self.current + 1] }
    }

    fn previous(&self) -> char {
        if self.current == 0 { '\0' } else { self.source[self.current - 1] }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }

    fn make_token(&self, kind: TokenKind) -> Token {
        let lexeme: String = self.source[self.start..self.current].iter().collect();
        Token::new(kind, lexeme, self.line, self.col - (self.current - self.start), self.start)
    }

    fn error_token(&self, message: &str) -> Token {
        Token::new(
            TokenKind::Error(message.to_string()),
            self.source[self.start..self.current].iter().collect(),
            self.line,
            self.col - (self.current - self.start),
            self.start,
        )
    }

    fn skip_whitespace(&mut self) {
        while !self.is_at_end() && (self.peek() == ' ' || self.peek() == '\t' || self.peek() == '\r') {
            self.advance();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(source: &str) -> Vec<Token> {
        Scanner::new(source).scan_all()
    }

    #[test]
    fn test_basic_tokens() {
        let tokens = scan("let x = 42");
        assert_eq!(tokens[0].kind, TokenKind::Keyword(Keyword::Let));
        assert_eq!(tokens[1].kind, TokenKind::Identifier("x".to_string()));
        assert_eq!(tokens[2].kind, TokenKind::Equal);
        assert_eq!(tokens[3].kind, TokenKind::Integer(42));
    }

    #[test]
    fn test_operators() {
        let tokens = scan("a + b * c ** d");
        assert_eq!(tokens[1].kind, TokenKind::Plus);
        assert_eq!(tokens[3].kind, TokenKind::Star);
        assert_eq!(tokens[5].kind, TokenKind::StarStar);
    }

    #[test]
    fn test_string_literal() {
        let tokens = scan(r#""hello world""#);
        assert_eq!(tokens[0].kind, TokenKind::String("hello world".to_string()));
    }

    #[test]
    fn test_string_escapes() {
        let tokens = scan(r#""hello\n\tworld""#);
        assert_eq!(tokens[0].kind, TokenKind::String("hello\n\tworld".to_string()));
    }

    #[test]
    fn test_float() {
        let tokens = scan("3.14");
        assert_eq!(tokens[0].kind, TokenKind::Float(3.14));
    }

    #[test]
    fn test_hex_number() {
        let tokens = scan("0xFF");
        assert_eq!(tokens[0].kind, TokenKind::Integer(255));
    }

    #[test]
    fn test_comparison() {
        let tokens = scan("a == b != c < d <= e > f >= g");
        assert_eq!(tokens[1].kind, TokenKind::EqualEqual);
        assert_eq!(tokens[3].kind, TokenKind::BangEqual);
        assert_eq!(tokens[5].kind, TokenKind::Less);
        assert_eq!(tokens[7].kind, TokenKind::LessEqual);
        assert_eq!(tokens[9].kind, TokenKind::Greater);
        assert_eq!(tokens[11].kind, TokenKind::GreaterEqual);
    }

    #[test]
    fn test_logical_operators() {
        let tokens = scan("a and b or not c");
        assert_eq!(tokens[1].kind, TokenKind::AmpAmp);
        assert_eq!(tokens[3].kind, TokenKind::PipePipe);
        assert_eq!(tokens[5].kind, TokenKind::Bang);
    }

    #[test]
    fn test_arrow_and_fat_arrow() {
        let tokens = scan("-> =>");
        assert_eq!(tokens[0].kind, TokenKind::Arrow);
        assert_eq!(tokens[1].kind, TokenKind::FatArrow);
    }

    #[test]
    fn test_range_operators() {
        let tokens = scan("0..10 0..=10 ...");
        assert_eq!(tokens[1].kind, TokenKind::DotDot);
        assert_eq!(tokens[4].kind, TokenKind::DotDotEqual);
        assert_eq!(tokens[7].kind, TokenKind::DotDotDot);
    }

    #[test]
    fn test_assignment_operators() {
        let tokens = scan("a += b -= c *= d /= e %= f &= g |= h ^= i <<= j >>= k **= l");
        assert_eq!(tokens[1].kind, TokenKind::PlusEqual);
        assert_eq!(tokens[3].kind, TokenKind::MinusEqual);
        assert_eq!(tokens[5].kind, TokenKind::StarEqual);
        assert_eq!(tokens[7].kind, TokenKind::SlashEqual);
        assert_eq!(tokens[9].kind, TokenKind::PercentEqual);
    }

    #[test]
    fn test_comments() {
        let tokens = scan("let x = 42 // this is a comment");
        assert_eq!(tokens[0].kind, TokenKind::Keyword(Keyword::Let));
    }
}
