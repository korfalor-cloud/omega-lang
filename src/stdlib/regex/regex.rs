use std::collections::HashMap;

pub struct OmegaRegex {
    pattern: String,
    compiled: Option<CompiledPattern>,
}

#[derive(Debug, Clone)]
struct CompiledPattern {
    instructions: Vec<RegexInstruction>,
    groups: usize,
}

#[derive(Debug, Clone)]
enum RegexInstruction {
    Char(char),
    AnyChar,
    CharClass(Vec<char>, bool), // chars, negated
    StartAnchor,
    EndAnchor,
    Quantifier(QuantifierType, usize), // type, instruction index
    GroupStart(usize), // group number
    GroupEnd(usize),
    Alternation(usize), // jump offset
    Jump(usize),
    Match,
}

#[derive(Debug, Clone)]
enum QuantifierType {
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
    Exactly(usize),
    AtLeast(usize),
    Range(usize, usize),
}

impl OmegaRegex {
    pub fn new(pattern: &str) -> Result<Self, String> {
        let compiled = Self::compile(pattern)?;
        Ok(Self {
            pattern: pattern.to_string(),
            compiled: Some(compiled),
        })
    }

    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    fn compile(pattern: &str) -> Result<CompiledPattern, String> {
        let mut instructions = Vec::new();
        let mut chars = pattern.chars().peekable();
        let mut groups = 0;

        while let Some(c) = chars.next() {
            match c {
                '.' => instructions.push(RegexInstruction::AnyChar),
                '^' => instructions.push(RegexInstruction::StartAnchor),
                '$' => instructions.push(RegexInstruction::EndAnchor),
                '*' => {
                    let idx = instructions.len() - 1;
                    instructions.push(RegexInstruction::Quantifier(QuantifierType::ZeroOrMore, idx));
                }
                '+' => {
                    let idx = instructions.len() - 1;
                    instructions.push(RegexInstruction::Quantifier(QuantifierType::OneOrMore, idx));
                }
                '?' => {
                    let idx = instructions.len() - 1;
                    instructions.push(RegexInstruction::Quantifier(QuantifierType::ZeroOrOne, idx));
                }
                '(' => {
                    groups += 1;
                    instructions.push(RegexInstruction::GroupStart(groups));
                }
                ')' => {
                    instructions.push(RegexInstruction::GroupEnd(groups));
                }
                '[' => {
                    let mut chars_in_class = Vec::new();
                    let negated = if chars.peek() == Some(&'^') {
                        chars.next();
                        true
                    } else {
                        false
                    };

                    while let Some(c) = chars.next() {
                        if c == ']' {
                            break;
                        }
                        if c == '\\' {
                            if let Some(escaped) = chars.next() {
                                chars_in_class.push(escaped);
                            }
                        } else {
                            chars_in_class.push(c);
                        }
                    }

                    instructions.push(RegexInstruction::CharClass(chars_in_class, negated));
                }
                '\\' => {
                    if let Some(escaped) = chars.next() {
                        match escaped {
                            'd' => {
                                instructions.push(RegexInstruction::CharClass(
                                    ('0'..='9').collect(),
                                    false,
                                ));
                            }
                            'D' => {
                                instructions.push(RegexInstruction::CharClass(
                                    ('0'..='9').collect(),
                                    true,
                                ));
                            }
                            'w' => {
                                let mut word_chars: Vec<char> = ('a'..='z').collect();
                                word_chars.extend('A'..='Z');
                                word_chars.extend('0'..='9');
                                word_chars.push('_');
                                instructions.push(RegexInstruction::CharClass(word_chars, false));
                            }
                            'W' => {
                                let mut word_chars: Vec<char> = ('a'..='z').collect();
                                word_chars.extend('A'..='Z');
                                word_chars.extend('0'..='9');
                                word_chars.push('_');
                                instructions.push(RegexInstruction::CharClass(word_chars, true));
                            }
                            's' => {
                                instructions.push(RegexInstruction::CharClass(
                                    vec![' ', '\t', '\n', '\r'],
                                    false,
                                ));
                            }
                            'S' => {
                                instructions.push(RegexInstruction::CharClass(
                                    vec![' ', '\t', '\n', '\r'],
                                    true,
                                ));
                            }
                            'n' => instructions.push(RegexInstruction::Char('\n')),
                            't' => instructions.push(RegexInstruction::Char('\t')),
                            'r' => instructions.push(RegexInstruction::Char('\r')),
                            _ => instructions.push(RegexInstruction::Char(escaped)),
                        }
                    }
                }
                '|' => {
                    instructions.push(RegexInstruction::Alternation(0));
                }
                _ => instructions.push(RegexInstruction::Char(c)),
            }
        }

        instructions.push(RegexInstruction::Match);

        Ok(CompiledPattern {
            instructions,
            groups,
        })
    }

    pub fn is_match(&self, text: &str) -> bool {
        if let Some(ref compiled) = self.compiled {
            self.try_match(text, compiled, 0, 0).is_some()
        } else {
            false
        }
    }

    pub fn find(&self, text: &str) -> Option<String> {
        if let Some(ref compiled) = self.compiled {
            for start in 0..text.len() {
                if let Some(end) = self.try_match(&text[start..], compiled, 0, 0) {
                    return Some(text[start..start + end].to_string());
                }
            }
        }
        None
    }

    pub fn find_all(&self, text: &str) -> Vec<String> {
        let mut results = Vec::new();
        let mut start = 0;

        while start < text.len() {
            if let Some(end) = self.find(&text[start..]) {
                results.push(end.clone());
                start += end.len();
            } else {
                break;
            }
        }

        results
    }

    pub fn replace(&self, text: &str, replacement: &str) -> String {
        let mut result = String::new();
        let mut last_end = 0;

        for start in 0..text.len() {
            if let Some(matched) = self.find(&text[start..]) {
                result.push_str(&text[last_end..start]);
                result.push_str(replacement);
                last_end = start + matched.len();
                start = last_end;
            }
        }

        result.push_str(&text[last_end..]);
        result
    }

    pub fn replace_all(&self, text: &str, replacement: &str) -> String {
        let mut result = text.to_string();
        while let Some(matched) = self.find(&result) {
            result = self.replace(&result, replacement);
        }
        result
    }

    pub fn split(&self, text: &str) -> Vec<String> {
        let mut results = Vec::new();
        let mut last_end = 0;

        for start in 0..text.len() {
            if let Some(matched) = self.find(&text[start..]) {
                results.push(text[last_end..start].to_string());
                last_end = start + matched.len();
            }
        }

        results.push(text[last_end..].to_string());
        results
    }

    pub fn groups(&self, text: &str) -> Option<Vec<String>> {
        // Simplified group extraction
        if let Some(matched) = self.find(text) {
            Some(vec![matched])
        } else {
            None
        }
    }

    fn try_match(&self, text: &str, compiled: &CompiledPattern, pc: usize, sp: usize) -> Option<usize> {
        if pc >= compiled.instructions.len() {
            return None;
        }

        let text_chars: Vec<char> = text.chars().collect();

        match &compiled.instructions[pc] {
            RegexInstruction::Char(expected) => {
                if sp < text_chars.len() && text_chars[sp] == *expected {
                    self.try_match(text, compiled, pc + 1, sp + 1)
                } else {
                    None
                }
            }
            RegexInstruction::AnyChar => {
                if sp < text_chars.len() {
                    self.try_match(text, compiled, pc + 1, sp + 1)
                } else {
                    None
                }
            }
            RegexInstruction::CharClass(chars, negated) => {
                if sp < text_chars.len() {
                    let in_class = chars.contains(&text_chars[sp]);
                    if in_class != *negated {
                        self.try_match(text, compiled, pc + 1, sp + 1)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            RegexInstruction::StartAnchor => {
                if sp == 0 {
                    self.try_match(text, compiled, pc + 1, sp)
                } else {
                    None
                }
            }
            RegexInstruction::EndAnchor => {
                if sp >= text_chars.len() {
                    self.try_match(text, compiled, pc + 1, sp)
                } else {
                    None
                }
            }
            RegexInstruction::Match => Some(sp),
            _ => None,
        }
    }

    // Common regex patterns
    pub fn email_pattern() -> Result<Self, String> {
        Self::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    }

    pub fn url_pattern() -> Result<Self, String> {
        Self::new(r"^https?://[^\s/$.?#].[^\s]*$")
    }

    pub fn ip_address_pattern() -> Result<Self, String> {
        Self::new(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    }

    pub fn phone_pattern() -> Result<Self, String> {
        Self::new(r"^\+?[\d\s-]{10,}$")
    }

    pub fn date_pattern() -> Result<Self, String> {
        Self::new(r"^\d{4}-\d{2}-\d{2}$")
    }

    pub fn time_pattern() -> Result<Self, String> {
        Self::new(r"^\d{2}:\d{2}(:\d{2})?$")
    }

    pub fn hex_color_pattern() -> Result<Self, String> {
        Self::new(r"^#[0-9a-fA-F]{6}$")
    }

    pub fn integer_pattern() -> Result<Self, String> {
        Self::new(r"^-?\d+$")
    }

    pub fn float_pattern() -> Result<Self, String> {
        Self::new(r"^-?\d+\.\d+$")
    }

    pub fn alphanumeric_pattern() -> Result<Self, String> {
        Self::new(r"^[a-zA-Z0-9]+$")
    }

    pub fn alpha_pattern() -> Result<Self, String> {
        Self::new(r"^[a-zA-Z]+$")
    }

    pub fn numeric_pattern() -> Result<Self, String> {
        Self::new(r"^\d+$")
    }
}

// Helper functions
pub fn is_email(s: &str) -> bool {
    OmegaRegex::email_pattern()
        .map(|r| r.is_match(s))
        .unwrap_or(false)
}

pub fn is_url(s: &str) -> bool {
    OmegaRegex::url_pattern()
        .map(|r| r.is_match(s))
        .unwrap_or(false)
}

pub fn is_ip_address(s: &str) -> bool {
    OmegaRegex::ip_address_pattern()
        .map(|r| r.is_match(s))
        .unwrap_or(false)
}

pub fn is_phone(s: &str) -> bool {
    OmegaRegex::phone_pattern()
        .map(|r| r.is_match(s))
        .unwrap_or(false)
}

pub fn is_date(s: &str) -> bool {
    OmegaRegex::date_pattern()
        .map(|r| r.is_match(s))
        .unwrap_or(false)
}

pub fn is_hex_color(s: &str) -> bool {
    OmegaRegex::hex_color_pattern()
        .map(|r| r.is_match(s))
        .unwrap_or(false)
}

pub fn is_integer(s: &str) -> bool {
    OmegaRegex::integer_pattern()
        .map(|r| r.is_match(s))
        .unwrap_or(false)
}

pub fn is_float(s: &str) -> bool {
    OmegaRegex::float_pattern()
        .map(|r| r.is_match(s))
        .unwrap_or(false)
}
