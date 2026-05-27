use crate::errors::{OmegaError, OmegaResult};
use crate::vm::stack::Value;

pub fn len(s: &str) -> usize {
    s.len()
}

pub fn char_len(s: &str) -> usize {
    s.chars().count()
}

pub fn is_empty(s: &str) -> bool {
    s.is_empty()
}

pub fn to_upper(s: &str) -> String {
    s.to_uppercase()
}

pub fn to_lower(s: &str) -> String {
    s.to_lowercase()
}

pub fn trim(s: &str) -> String {
    s.trim().to_string()
}

pub fn trim_start(s: &str) -> String {
    s.trim_start().to_string()
}

pub fn trim_end(s: &str) -> String {
    s.trim_end().to_string()
}

pub fn trim_matches(s: &str, pattern: &str) -> String {
    s.trim_matches(pattern.chars().next().unwrap_or(' ')).to_string()
}

pub fn starts_with(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

pub fn ends_with(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

pub fn contains(s: &str, pattern: &str) -> bool {
    s.contains(pattern)
}

pub fn find(s: &str, pattern: &str) -> Option<usize> {
    s.find(pattern)
}

pub fn rfind(s: &str, pattern: &str) -> Option<usize> {
    s.rfind(pattern)
}

pub fn replace(s: &str, from: &str, to: &str) -> String {
    s.replace(from, to)
}

pub fn replace_n(s: &str, from: &str, to: &str, count: usize) -> String {
    let mut result = s.to_string();
    for _ in 0..count {
        if let Some(pos) = result.find(from) {
            result = format!("{}{}{}", &result[..pos], to, &result[pos + from.len()..]);
        } else {
            break;
        }
    }
    result
}

pub fn split(s: &str, delimiter: &str) -> Vec<String> {
    s.split(delimiter).map(String::from).collect()
}

pub fn split_whitespace(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

pub fn split_lines(s: &str) -> Vec<String> {
    s.lines().map(String::from).collect()
}

pub fn join(strings: &[&str], separator: &str) -> String {
    strings.join(separator)
}

pub fn repeat(s: &str, count: usize) -> String {
    s.repeat(count)
}

pub fn chars(s: &str) -> Vec<char> {
    s.chars().collect()
}

pub fn bytes(s: &str) -> Vec<u8> {
    s.bytes().collect()
}

pub fn char_at(s: &str, index: usize) -> OmegaResult<char> {
    s.chars().nth(index).ok_or_else(|| OmegaError::IndexOutOfBounds {
        index: index as i64,
        length: s.chars().count(),
    })
}

pub fn substring(s: &str, start: usize, end: usize) -> OmegaResult<String> {
    let chars: Vec<char> = s.chars().collect();
    if start > end || end > chars.len() {
        return Err(OmegaError::IndexOutOfBounds {
            index: end as i64,
            length: chars.len(),
        });
    }
    Ok(chars[start..end].iter().collect())
}

pub fn pad_left(s: &str, width: usize, fill: char) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        let padding: String = std::iter::repeat(fill).take(width - len).collect();
        format!("{}{}", padding, s)
    }
}

pub fn pad_right(s: &str, width: usize, fill: char) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        let padding: String = std::iter::repeat(fill).take(width - len).collect();
        format!("{}{}", s, padding)
    }
}

pub fn center(s: &str, width: usize, fill: char) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        let left_pad = (width - len) / 2;
        let right_pad = width - len - left_pad;
        let left: String = std::iter::repeat(fill).take(left_pad).collect();
        let right: String = std::iter::repeat(fill).take(right_pad).collect();
        format!("{}{}{}", left, s, right)
    }
}

pub fn reverse(s: &str) -> String {
    s.chars().rev().collect()
}

pub fn is_alpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphabetic())
}

pub fn is_numeric(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_numeric())
}

pub fn is_alphanumeric(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric())
}

pub fn is_whitespace(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_whitespace())
}

pub fn is_uppercase(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| !c.is_alphabetic() || c.is_uppercase())
}

pub fn is_lowercase(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| !c.is_alphabetic() || c.is_lowercase())
}

pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + &chars.as_str().to_lowercase(),
    }
}

pub fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| capitalize(word))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn camel_case(s: &str) -> String {
    let words: Vec<&str> = s.split(|c: char| c == '_' || c == '-' || c.is_whitespace()).collect();
    let mut result = String::new();
    for (i, word) in words.iter().enumerate() {
        if i == 0 {
            result.push_str(&word.to_lowercase());
        } else {
            result.push_str(&capitalize(word));
        }
    }
    result
}

pub fn snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(c.to_lowercase().next().unwrap_or(c));
    }
    result
}

pub fn kebab_case(s: &str) -> String {
    snake_case(s).replace('_', "-")
}

pub fn count(s: &str, pattern: &str) -> usize {
    s.matches(pattern).count()
}

pub fn index_of(s: &str, pattern: &str) -> Option<usize> {
    s.find(pattern)
}

pub fn last_index_of(s: &str, pattern: &str) -> Option<usize> {
    s.rfind(pattern)
}

pub fn slice(s: &str, start: i64, end: i64) -> OmegaResult<String> {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len() as i64;
    let start = if start < 0 { len + start } else { start };
    let end = if end < 0 { len + end } else { end };
    let start = start.max(0) as usize;
    let end = end.min(len) as usize;
    if start > end {
        return Ok(String::new());
    }
    Ok(chars[start..end].iter().collect())
}

pub fn insert(s: &str, index: usize, text: &str) -> OmegaResult<String> {
    let chars: Vec<char> = s.chars().collect();
    if index > chars.len() {
        return Err(OmegaError::IndexOutOfBounds {
            index: index as i64,
            length: chars.len(),
        });
    }
    let mut result: String = chars[..index].iter().collect();
    result.push_str(text);
    result.push_str(&chars[index..].iter().collect::<String>());
    Ok(result)
}

pub fn remove(s: &str, start: usize, count: usize) -> OmegaResult<String> {
    let chars: Vec<char> = s.chars().collect();
    if start > chars.len() {
        return Err(OmegaError::IndexOutOfBounds {
            index: start as i64,
            length: chars.len(),
        });
    }
    let end = (start + count).min(chars.len());
    let mut result: String = chars[..start].iter().collect();
    result.push_str(&chars[end..].iter().collect::<String>());
    Ok(result)
}

pub fn matches(s: &str, pattern: &str) -> bool {
    s.contains(pattern)
}

pub fn escape(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            '\n' => result.push_str("\\n"),
            '\t' => result.push_str("\\t"),
            '\r' => result.push_str("\\r"),
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\'' => result.push_str("\\'"),
            '\0' => result.push_str("\\0"),
            _ => result.push(c),
        }
    }
    result
}

pub fn unescape(s: &str) -> OmegaResult<String> {
    let mut result = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('\'') => result.push('\''),
                Some('0') => result.push('\0'),
                Some(c) => result.push(c),
                None => return Err(OmegaError::ValueError {
                    message: "Unexpected end of escape sequence".to_string(),
                }),
            }
        } else {
            result.push(c);
        }
    }
    Ok(result)
}

pub fn to_string(value: &Value) -> String {
    value.format_display()
}

pub fn parse_int(s: &str) -> OmegaResult<i64> {
    s.trim().parse().map_err(|_| OmegaError::ValueError {
        message: format!("Cannot parse '{}' as integer", s),
    })
}

pub fn parse_float(s: &str) -> OmegaResult<f64> {
    s.trim().parse().map_err(|_| OmegaError::ValueError {
        message: format!("Cannot parse '{}' as float", s),
    })
}

pub fn from_chars(chars: &[char]) -> String {
    chars.iter().collect()
}

pub fn from_bytes(bytes: &[u8]) -> OmegaResult<String> {
    String::from_utf8(bytes.to_vec()).map_err(|e| OmegaError::EncodingError {
        message: e.to_string(),
    })
}

pub fn to_utf8(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

pub fn char_code(c: char) -> u32 {
    c as u32
}

pub fn from_char_code(code: u32) -> OmegaResult<char> {
    char::from_u32(code).ok_or_else(|| OmegaError::ValueError {
        message: format!("Invalid char code: {}", code),
    })
}

pub fn graphemes(s: &str) -> Vec<String> {
    use unicode_segmentation::UnicodeSegmentation;
    s.graphemes(true).map(String::from).collect()
}

pub fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

pub fn sentence_count(s: &str) -> usize {
    s.split(|c: char| c == '.' || c == '!' || c == '?')
        .filter(|s| !s.trim().is_empty())
        .count()
}

pub fn line_count(s: &str) -> usize {
    s.lines().count()
}

pub fn truncate(s: &str, max_len: usize, suffix: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_len {
        s.to_string()
    } else {
        let truncated: String = chars[..max_len.saturating_sub(suffix.chars().count())].iter().collect();
        format!("{}{}", truncated, suffix)
    }
}

pub fn wrap(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();
    for word in s.split_whitespace() {
        if current_line.is_empty() {
            current_line.push_str(word);
        } else if current_line.len() + 1 + word.len() <= width {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(current_line);
            current_line = word.to_string();
        }
    }
    if !current_line.is_empty() {
        lines.push(current_line);
    }
    lines
}

pub fn indent(s: &str, prefix: &str) -> String {
    s.lines()
        .map(|line| format!("{}{}", prefix, line))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn dedent(s: &str) -> String {
    let min_indent = s.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    s.lines()
        .map(|line| {
            if line.len() >= min_indent {
                &line[min_indent..]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
