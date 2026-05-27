/// Parser combinator library for building parsers.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub position: usize,
    pub expected: Vec<String>,
    pub found: Option<char>,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse error at position {}: expected {:?}", self.position, self.expected)?;
        if let Some(c) = self.found {
            write!(f, ", found '{}'", c)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ParseResult<'a, T> {
    pub value: T,
    pub remaining: &'a str,
    pub position: usize,
}

pub type ParserOutput<'a, T> = Result<ParseResult<'a, T>, ParseError>;

pub trait Parser<'a, T>: Clone {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, T>;

    fn map<U, F: Fn(T) -> U + Clone>(self, f: F) -> Map<Self, F>
    where Self: Sized {
        Map { parser: self, f }
    }

    fn and_then<U, P2, F>(self, f: F) -> AndThen<Self, F>
    where
        Self: Sized,
        P2: Parser<'a, U>,
        F: Fn(T) -> P2 + Clone,
    {
        AndThen { parser: self, f }
    }

    fn then<U, P2: Parser<'a, U>>(self, other: P2) -> Then<Self, P2> {
        Then { first: self, second: other }
    }

    fn skip<U, P2: Parser<'a, U>>(self, other: P2) -> Skip<Self, P2> {
        Skip { first: self, second: other }
    }

    fn preceeds<U, P2: Parser<'a, U>>(self, other: P2) -> Preceeds<Self, P2> {
        Preceeds { first: self, second: other }
    }

    fn or<P2: Parser<'a, T>>(self, other: P2) -> Or<Self, P2> {
        Or { first: self, second: other }
    }

    fn repeated(self) -> Repeated<Self> where Self: Sized {
        Repeated { parser: self }
    }

    fn at_least(self, min: usize) -> AtLeast<Self> where Self: Sized {
        AtLeast { parser: self, min }
    }

    fn at_most(self, max: usize) -> AtMost<Self> where Self: Sized {
        AtMost { parser: self, max }
    }

    fn separated_by<S, P2: Parser<'a, S>>(self, separator: P2) -> SeparatedBy<Self, P2, S>
    where Self: Sized {
        SeparatedBy { parser: self, separator, _phantom: std::marker::PhantomData }
    }

    fn delimited<L, R, LP: Parser<'a, L>, RP: Parser<'a, R>>(
        self, left: LP, right: RP,
    ) -> Delimited<Self, LP, RP, L, R> where Self: Sized {
        Delimited { parser: self, left, right, _phantom: std::marker::PhantomData }
    }

    fn padded(self) -> Padded<Self> where Self: Sized {
        Padded { parser: self }
    }

    fn attempt(self) -> Attempt<Self> where Self: Sized {
        Attempt { parser: self }
    }

    fn label(self, label: &str) -> Label<Self> where Self: Sized {
        Label { parser: self, label: label.to_string() }
    }
}

// Combinator structs

#[derive(Clone)]
pub struct Map<P, F> {
    parser: P,
    f: F,
}

impl<'a, T, U, P: Parser<'a, T>, F: Fn(T) -> U + Clone> Parser<'a, U> for Map<P, F> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, U> {
        self.parser.parse(input, position).map(|result| ParseResult {
            value: (self.f)(result.value),
            remaining: result.remaining,
            position: result.position,
        })
    }
}

#[derive(Clone)]
pub struct AndThen<P, F> {
    parser: P,
    f: F,
}

impl<'a, T, U, P: Parser<'a, T>, P2: Parser<'a, U>, F: Fn(T) -> P2 + Clone> Parser<'a, U> for AndThen<P, F> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, U> {
        let result = self.parser.parse(input, position)?;
        let next_parser = (self.f)(result.value);
        next_parser.parse(result.remaining, result.position)
    }
}

#[derive(Clone)]
pub struct Then<P1, P2> {
    first: P1,
    second: P2,
}

impl<'a, T, U, P1: Parser<'a, T>, P2: Parser<'a, U>> Parser<'a, (T, U)> for Then<P1, P2> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, (T, U)> {
        let r1 = self.first.parse(input, position)?;
        let r2 = self.second.parse(r1.remaining, r1.position)?;
        Ok(ParseResult {
            value: (r1.value, r2.value),
            remaining: r2.remaining,
            position: r2.position,
        })
    }
}

#[derive(Clone)]
pub struct Skip<P1, P2> {
    first: P1,
    second: P2,
}

impl<'a, T, U, P1: Parser<'a, T>, P2: Parser<'a, U>> Parser<'a, T> for Skip<P1, P2> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, T> {
        let r1 = self.first.parse(input, position)?;
        let r2 = self.second.parse(r1.remaining, r1.position)?;
        Ok(ParseResult {
            value: r1.value,
            remaining: r2.remaining,
            position: r2.position,
        })
    }
}

#[derive(Clone)]
pub struct Preceeds<P1, P2> {
    first: P1,
    second: P2,
}

impl<'a, T, U, P1: Parser<'a, T>, P2: Parser<'a, U>> Parser<'a, U> for Preceeds<P1, P2> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, U> {
        let r1 = self.first.parse(input, position)?;
        self.second.parse(r1.remaining, r1.position)
    }
}

#[derive(Clone)]
pub struct Or<P1, P2> {
    first: P1,
    second: P2,
}

impl<'a, T, P1: Parser<'a, T>, P2: Parser<'a, T>> Parser<'a, T> for Or<P1, P2> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, T> {
        match self.first.parse(input, position) {
            ok @ Ok(_) => ok,
            Err(e1) => match self.second.parse(input, position) {
                ok @ Ok(_) => ok,
                Err(e2) => {
                    let mut expected = e1.expected;
                    expected.extend(e2.expected);
                    Err(ParseError { position, expected, found: e2.found })
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Repeated<P> {
    parser: P,
}

impl<'a, T, P: Parser<'a, T>> Parser<'a, Vec<T>> for Repeated<P> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, Vec<T>> {
        let mut results = Vec::new();
        let mut remaining = input;
        let mut pos = position;

        loop {
            match self.parser.parse(remaining, pos) {
                Ok(result) => {
                    results.push(result.value);
                    remaining = result.remaining;
                    pos = result.position;
                }
                Err(_) => break,
            }
        }

        Ok(ParseResult { value: results, remaining, position: pos })
    }
}

#[derive(Clone)]
pub struct AtLeast<P> {
    parser: P,
    min: usize,
}

impl<'a, T, P: Parser<'a, T>> Parser<'a, Vec<T>> for AtLeast<P> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, Vec<T>> {
        let mut results = Vec::new();
        let mut remaining = input;
        let mut pos = position;

        loop {
            match self.parser.parse(remaining, pos) {
                Ok(result) => {
                    results.push(result.value);
                    remaining = result.remaining;
                    pos = result.position;
                }
                Err(_) => break,
            }
        }

        if results.len() >= self.min {
            Ok(ParseResult { value: results, remaining, position: pos })
        } else {
            Err(ParseError {
                position: pos,
                expected: vec![format!("at least {} matches", self.min)],
                found: None,
            })
        }
    }
}

#[derive(Clone)]
pub struct AtMost<P> {
    parser: P,
    max: usize,
}

impl<'a, T, P: Parser<'a, T>> Parser<'a, Vec<T>> for AtMost<P> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, Vec<T>> {
        let mut results = Vec::new();
        let mut remaining = input;
        let mut pos = position;

        while results.len() < self.max {
            match self.parser.parse(remaining, pos) {
                Ok(result) => {
                    results.push(result.value);
                    remaining = result.remaining;
                    pos = result.position;
                }
                Err(_) => break,
            }
        }

        Ok(ParseResult { value: results, remaining, position: pos })
    }
}

#[derive(Clone)]
pub struct SeparatedBy<P, S, T> {
    parser: P,
    separator: S,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, S, P: Parser<'a, T>, SP: Parser<'a, S>> Parser<'a, Vec<T>> for SeparatedBy<P, SP, T> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, Vec<T>> {
        let mut results = Vec::new();
        let mut remaining = input;
        let mut pos = position;

        // Parse first element
        match self.parser.parse(remaining, pos) {
            Ok(result) => {
                results.push(result.value);
                remaining = result.remaining;
                pos = result.position;
            }
            Err(_) => {
                return Ok(ParseResult { value: results, remaining, position: pos });
            }
        }

        // Parse remaining elements with separator
        loop {
            match self.separator.parse(remaining, pos) {
                Ok(sep_result) => {
                    match self.parser.parse(sep_result.remaining, sep_result.position) {
                        Ok(result) => {
                            results.push(result.value);
                            remaining = result.remaining;
                            pos = result.position;
                        }
                        Err(_) => break,
                    }
                }
                Err(_) => break,
            }
        }

        Ok(ParseResult { value: results, remaining, position: pos })
    }
}

#[derive(Clone)]
pub struct Delimited<P, LP, RP, L, R> {
    parser: P,
    left: LP,
    right: RP,
    _phantom: std::marker::PhantomData<(L, R)>,
}

impl<'a, T, L, R, P: Parser<'a, T>, LP: Parser<'a, L>, RP: Parser<'a, R>> Parser<'a, T>
    for Delimited<P, LP, RP, L, R>
{
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, T> {
        let left_result = self.left.parse(input, position)?;
        let value_result = self.parser.parse(left_result.remaining, left_result.position)?;
        let right_result = self.right.parse(value_result.remaining, value_result.position)?;
        Ok(ParseResult {
            value: value_result.value,
            remaining: right_result.remaining,
            position: right_result.position,
        })
    }
}

#[derive(Clone)]
pub struct Padded<P> {
    parser: P,
}

impl<'a, T, P: Parser<'a, T>> Parser<'a, T> for Padded<P> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, T> {
        let trimmed = &input[position..];
        let trimmed_input = trimmed.trim_start();
        let offset = input.len() - trimmed_input.len();
        self.parser.parse(trimmed_input, position + offset)
    }
}

#[derive(Clone)]
pub struct Attempt<P> {
    parser: P,
}

impl<'a, T, P: Parser<'a, T>> Parser<'a, T> for Attempt<P> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, T> {
        self.parser.parse(input, position)
    }
}

#[derive(Clone)]
pub struct Label<P> {
    parser: P,
    label: String,
}

impl<'a, T, P: Parser<'a, T>> Parser<'a, T> for Label<P> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, T> {
        self.parser.parse(input, position).map_err(|e| ParseError {
            position: e.position,
            expected: vec![self.label.clone()],
            found: e.found,
        })
    }
}

// Primitive parsers

#[derive(Clone)]
pub struct CharParser {
    target: char,
}

pub fn char(c: char) -> CharParser {
    CharParser { target: c }
}

impl<'a> Parser<'a, char> for CharParser {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, char> {
        match input.chars().next() {
            Some(c) if c == self.target => Ok(ParseResult {
                value: c,
                remaining: &input[c.len_utf8()..],
                position: position + 1,
            }),
            other => Err(ParseError {
                position,
                expected: vec![format!("'{}'", self.target)],
                found: other,
            }),
        }
    }
}

#[derive(Clone)]
pub struct AnyCharParser;

pub fn any_char() -> AnyCharParser {
    AnyCharParser
}

impl<'a> Parser<'a, char> for AnyCharParser {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, char> {
        match input.chars().next() {
            Some(c) => Ok(ParseResult {
                value: c,
                remaining: &input[c.len_utf8()..],
                position: position + 1,
            }),
            None => Err(ParseError {
                position,
                expected: vec!["any character".to_string()],
                found: None,
            }),
        }
    }
}

#[derive(Clone)]
pub struct CharPredParser<F> {
    predicate: F,
    description: String,
}

pub fn char_pred<F: Fn(char) -> bool + Clone>(predicate: F, description: &str) -> CharPredParser<F> {
    CharPredParser {
        predicate,
        description: description.to_string(),
    }
}

impl<'a, F: Fn(char) -> bool + Clone> Parser<'a, char> for CharPredParser<F> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, char> {
        match input.chars().next() {
            Some(c) if (self.predicate)(c) => Ok(ParseResult {
                value: c,
                remaining: &input[c.len_utf8()..],
                position: position + 1,
            }),
            other => Err(ParseError {
                position,
                expected: vec![self.description.clone()],
                found: other,
            }),
        }
    }
}

#[derive(Clone)]
pub struct StringParser {
    target: String,
}

pub fn string(s: &str) -> StringParser {
    StringParser { target: s.to_string() }
}

impl<'a> Parser<'a, &'a str> for StringParser {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, &'a str> {
        if input.starts_with(&self.target) {
            Ok(ParseResult {
                value: &input[..self.target.len()],
                remaining: &input[self.target.len()..],
                position: position + self.target.len(),
            })
        } else {
            Err(ParseError {
                position,
                expected: vec![format!("\"{}\"", self.target)],
                found: input.chars().next(),
            })
        }
    }
}

#[derive(Clone)]
pub struct DigitParser;

pub fn digit() -> DigitParser {
    DigitParser
}

impl<'a> Parser<'a, char> for DigitParser {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, char> {
        match input.chars().next() {
            Some(c) if c.is_ascii_digit() => Ok(ParseResult {
                value: c,
                remaining: &input[1..],
                position: position + 1,
            }),
            other => Err(ParseError {
                position,
                expected: vec!["digit".to_string()],
                found: other,
            }),
        }
    }
}

#[derive(Clone)]
pub struct WhitespaceParser;

pub fn whitespace() -> WhitespaceParser {
    WhitespaceParser
}

impl<'a> Parser<'a, &'a str> for WhitespaceParser {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, &'a str> {
        let end = input.chars().take_while(|c| c.is_whitespace()).count();
        if end > 0 {
            Ok(ParseResult {
                value: &input[..end],
                remaining: &input[end..],
                position: position + end,
            })
        } else {
            Ok(ParseResult {
                value: "",
                remaining: input,
                position,
            })
        }
    }
}

#[derive(Clone)]
pub struct IntParser;

pub fn integer() -> IntParser {
    IntParser
}

impl<'a> Parser<'a, i64> for IntParser {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, i64> {
        let mut end = 0;
        let chars: Vec<char> = input.chars().collect();

        if chars.get(0) == Some(&'-') || chars.get(0) == Some(&'+') {
            end = 1;
        }

        while end < chars.len() && chars[end].is_ascii_digit() {
            end += 1;
        }

        if end == 0 || (end == 1 && (chars[0] == '-' || chars[0] == '+')) {
            return Err(ParseError {
                position,
                expected: vec!["integer".to_string()],
                found: chars.get(0).copied(),
            });
        }

        let s = &input[..end];
        match s.parse::<i64>() {
            Ok(value) => Ok(ParseResult {
                value,
                remaining: &input[end..],
                position: position + end,
            }),
            Err(_) => Err(ParseError {
                position,
                expected: vec!["integer".to_string()],
                found: chars.get(0).copied(),
            }),
        }
    }
}

#[derive(Clone)]
pub struct FloatParser;

pub fn float() -> FloatParser {
    FloatParser
}

impl<'a> Parser<'a, f64> for FloatParser {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, f64> {
        let mut end = 0;
        let chars: Vec<char> = input.chars().collect();

        if chars.get(0) == Some(&'-') || chars.get(0) == Some(&'+') {
            end = 1;
        }

        while end < chars.len() && chars[end].is_ascii_digit() {
            end += 1;
        }

        if end < chars.len() && chars[end] == '.' {
            end += 1;
            while end < chars.len() && chars[end].is_ascii_digit() {
                end += 1;
            }
        }

        if end < chars.len() && (chars[end] == 'e' || chars[end] == 'E') {
            end += 1;
            if end < chars.len() && (chars[end] == '+' || chars[end] == '-') {
                end += 1;
            }
            while end < chars.len() && chars[end].is_ascii_digit() {
                end += 1;
            }
        }

        if end == 0 {
            return Err(ParseError {
                position,
                expected: vec!["float".to_string()],
                found: chars.get(0).copied(),
            });
        }

        let s = &input[..end];
        match s.parse::<f64>() {
            Ok(value) => Ok(ParseResult {
                value,
                remaining: &input[end..],
                position: position + end,
            }),
            Err(_) => Err(ParseError {
                position,
                expected: vec!["float".to_string()],
                found: chars.get(0).copied(),
            }),
        }
    }
}

#[derive(Clone)]
pub struct EofParser;

pub fn eof() -> EofParser {
    EofParser
}

impl<'a> Parser<'a, ()> for EofParser {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, ()> {
        if input.is_empty() {
            Ok(ParseResult { value: (), remaining: "", position })
        } else {
            Err(ParseError {
                position,
                expected: vec!["end of input".to_string()],
                found: input.chars().next(),
            })
        }
    }
}

pub fn succeed<'a, T: Clone>(value: T) -> SucceedParser<T> {
    SucceedParser { value }
}

#[derive(Clone)]
pub struct SucceedParser<T> {
    value: T,
}

impl<'a, T: Clone> Parser<'a, T> for SucceedParser<T> {
    fn parse(&self, input: &'a str, position: usize) -> ParserOutput<'a, T> {
        Ok(ParseResult {
            value: self.value.clone(),
            remaining: input,
            position,
        })
    }
}

pub fn fail<'a, T>(message: &str) -> FailParser<T> {
    FailParser { message: message.to_string(), _phantom: std::marker::PhantomData }
}

#[derive(Clone)]
pub struct FailParser<T> {
    message: String,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> Parser<'a, T> for FailParser<T> {
    fn parse(&self, _input: &'a str, position: usize) -> ParserOutput<'a, T> {
        Err(ParseError {
            position,
            expected: vec![self.message.clone()],
            found: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_parser() {
        let parser = char('a');
        let result = parser.parse("abc", 0).unwrap();
        assert_eq!(result.value, 'a');
        assert_eq!(result.remaining, "bc");
    }

    #[test]
    fn test_string_parser() {
        let parser = string("hello");
        let result = parser.parse("hello world", 0).unwrap();
        assert_eq!(result.value, "hello");
        assert_eq!(result.remaining, " world");
    }

    #[test]
    fn test_then() {
        let parser = char('a').then(char('b'));
        let result = parser.parse("abc", 0).unwrap();
        assert_eq!(result.value, ('a', 'b'));
    }

    #[test]
    fn test_skip() {
        let parser = char('a').skip(char('b'));
        let result = parser.parse("abc", 0).unwrap();
        assert_eq!(result.value, 'a');
        assert_eq!(result.remaining, "c");
    }

    #[test]
    fn test_or() {
        let parser = char('a').or(char('b'));
        let r1 = parser.parse("abc", 0).unwrap();
        assert_eq!(r1.value, 'a');
        let r2 = parser.parse("bac", 0).unwrap();
        assert_eq!(r2.value, 'b');
    }

    #[test]
    fn test_repeated() {
        let parser = char('a').repeated();
        let result = parser.parse("aaabc", 0).unwrap();
        assert_eq!(result.value, vec!['a', 'a', 'a']);
    }

    #[test]
    fn test_separated_by() {
        let parser = digit().map(|c| c.to_digit(10).unwrap() as i64)
            .separated_by(char(','));
        let result = parser.parse("1,2,3", 0).unwrap();
        assert_eq!(result.value, vec![1, 2, 3]);
    }

    #[test]
    fn test_integer() {
        let result = integer().parse("42abc", 0).unwrap();
        assert_eq!(result.value, 42);

        let result = integer().parse("-7xyz", 0).unwrap();
        assert_eq!(result.value, -7);
    }

    #[test]
    fn test_float() {
        let result = float().parse("3.14abc", 0).unwrap();
        assert!((result.value - 3.14).abs() < 1e-10);

        let result = float().parse("1e5abc", 0).unwrap();
        assert!((result.value - 100000.0).abs() < 1e-10);
    }

    #[test]
    fn test_delimited() {
        let parser = integer().delimited(char('('), char(')'));
        let result = parser.parse("(42)abc", 0).unwrap();
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_map() {
        let parser = digit().map(|c| c.to_digit(10).unwrap());
        let result = parser.parse("5abc", 0).unwrap();
        assert_eq!(result.value, 5);
    }

    #[test]
    fn test_and_then() {
        let parser = digit()
            .map(|c| c.to_digit(10).unwrap() as usize)
            .and_then(|n| char_repeated('a', n));
        let result = parser.parse("3aaa", 0).unwrap();
        assert_eq!(result.value, "aaa");
    }

    fn char_repeated<'a>(c: char, n: usize) -> impl Parser<'a, &'a str> {
        move |input: &'a str, position: usize| -> ParserOutput<'a, &'a str> {
            let end = input.chars().take_while(|&ch| ch == c).count();
            if end >= n {
                Ok(ParseResult {
                    value: &input[..n],
                    remaining: &input[n..],
                    position: position + n,
                })
            } else {
                Err(ParseError {
                    position,
                    expected: vec![format!("{} '{}'", n, c)],
                    found: input.chars().next(),
                })
            }
        }
    }

    #[test]
    fn test_error() {
        let result = char('a').parse("bcd", 0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.position, 0);
        assert_eq!(err.found, Some('b'));
    }
}
