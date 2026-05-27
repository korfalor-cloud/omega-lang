/// Simple XML parser.

use super::dom::XmlNode;
use std::collections::HashMap;

#[derive(Debug)]
pub struct XmlParser {
    input: Vec<char>,
    pos: usize,
}

#[derive(Debug)]
pub struct ParseXmlError {
    pub position: usize,
    pub message: String,
}

impl std::fmt::Display for ParseXmlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XML parse error at position {}: {}", self.position, self.message)
    }
}

impl XmlParser {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    pub fn parse(&mut self) -> Result<XmlNode, ParseXmlError> {
        self.skip_whitespace();
        self.skip_prolog();
        self.skip_whitespace();
        self.parse_element()
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.input.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn expect(&mut self, expected: char) -> Result<(), ParseXmlError> {
        match self.advance() {
            Some(c) if c == expected => Ok(()),
            other => Err(ParseXmlError {
                position: self.pos,
                message: format!("Expected '{}', found {:?}", expected, other),
            }),
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_prolog(&mut self) {
        if self.pos + 1 < self.input.len() && self.input[self.pos] == '<' && self.input[self.pos + 1] == '?' {
            self.advance(); // <
            self.advance(); // ?
            while let Some(c) = self.advance() {
                if c == '?' && self.peek() == Some('>') {
                    self.advance();
                    break;
                }
            }
        }
    }

    fn parse_element(&mut self) -> Result<XmlNode, ParseXmlError> {
        self.expect('<')?;

        // Check for comment
        if self.pos + 2 < self.input.len()
            && self.input[self.pos] == '!'
            && self.input[self.pos + 1] == '-'
            && self.input[self.pos + 2] == '-'
        {
            self.advance(); // !
            self.advance(); // -
            self.advance(); // -
            let mut comment = String::new();
            loop {
                match self.advance() {
                    Some('-') if self.peek() == Some('-') => {
                        self.advance();
                        self.expect('>')?;
                        break;
                    }
                    Some(c) => comment.push(c),
                    None => return Err(ParseXmlError {
                        position: self.pos,
                        message: "Unterminated comment".to_string(),
                    }),
                }
            }
            return Ok(XmlNode::Comment(comment));
        }

        // Check for CDATA
        if self.pos + 7 < self.input.len()
            && self.input[self.pos] == '!'
            && self.input[self.pos + 1] == '['
            && self.input[self.pos + 2] == 'C'
            && self.input[self.pos + 3] == 'D'
            && self.input[self.pos + 4] == 'A'
            && self.input[self.pos + 5] == 'T'
            && self.input[self.pos + 6] == 'A'
            && self.input[self.pos + 7] == '['
        {
            for _ in 0..8 { self.advance(); }
            let mut cdata = String::new();
            loop {
                match self.advance() {
                    Some(']') if self.pos + 1 < self.input.len()
                        && self.input[self.pos] == ']'
                        && self.input[self.pos + 1] == '>' =>
                    {
                        self.advance();
                        self.advance();
                        break;
                    }
                    Some(c) => cdata.push(c),
                    None => return Err(ParseXmlError {
                        position: self.pos,
                        message: "Unterminated CDATA".to_string(),
                    }),
                }
            }
            return Ok(XmlNode::CData(cdata));
        }

        let tag = self.parse_tag_name()?;
        self.skip_whitespace();

        let mut attributes = HashMap::new();
        while let Some(c) = self.peek() {
            if c == '>' || c == '/' {
                break;
            }
            let (key, value) = self.parse_attribute()?;
            attributes.insert(key, value);
            self.skip_whitespace();
        }

        // Self-closing tag
        if self.peek() == Some('/') {
            self.advance();
            self.expect('>')?;
            return Ok(XmlNode::Element {
                tag,
                attributes,
                children: Vec::new(),
            });
        }

        self.expect('>')?;

        // Parse children
        let mut children = Vec::new();
        loop {
            self.skip_whitespace();
            if self.pos + 1 < self.input.len()
                && self.input[self.pos] == '<'
                && self.input[self.pos + 1] == '/'
            {
                // Closing tag
                self.advance();
                self.advance();
                let close_tag = self.parse_tag_name()?;
                if close_tag != tag {
                    return Err(ParseXmlError {
                        position: self.pos,
                        message: format!("Mismatched closing tag: expected '{}', found '{}'", tag, close_tag),
                    });
                }
                self.skip_whitespace();
                self.expect('>')?;
                break;
            }

            if self.pos >= self.input.len() {
                return Err(ParseXmlError {
                    position: self.pos,
                    message: format!("Unexpected end of input, expected closing tag '{}'", tag),
                });
            }

            if self.peek() == Some('<') {
                children.push(self.parse_element()?);
            } else {
                children.push(self.parse_text()?);
            }
        }

        Ok(XmlNode::Element {
            tag,
            attributes,
            children,
        })
    }

    fn parse_tag_name(&mut self) -> Result<String, ParseXmlError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' || c == '-' || c == ':' || c == '.' {
                self.advance();
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(ParseXmlError {
                position: self.pos,
                message: "Expected tag name".to_string(),
            });
        }
        Ok(self.input[start..self.pos].iter().collect())
    }

    fn parse_attribute(&mut self) -> Result<(String, String), ParseXmlError> {
        let key = self.parse_tag_name()?;
        self.skip_whitespace();
        self.expect('=')?;
        self.skip_whitespace();
        let value = self.parse_quoted_string()?;
        Ok((key, value))
    }

    fn parse_quoted_string(&mut self) -> Result<String, ParseXmlError> {
        let quote = self.advance().ok_or(ParseXmlError {
            position: self.pos,
            message: "Expected quote".to_string(),
        })?;

        if quote != '"' && quote != '\'' {
            return Err(ParseXmlError {
                position: self.pos,
                message: format!("Expected quote, found '{}'", quote),
            });
        }

        let mut result = String::new();
        loop {
            match self.advance() {
                Some(c) if c == quote => break,
                Some(c) => result.push(c),
                None => return Err(ParseXmlError {
                    position: self.pos,
                    message: "Unterminated string".to_string(),
                }),
            }
        }

        Ok(unescape_xml(&result))
    }

    fn parse_text(&mut self) -> Result<XmlNode, ParseXmlError> {
        let mut text = String::new();
        while let Some(c) = self.peek() {
            if c == '<' {
                break;
            }
            text.push(c);
            self.advance();
        }
        Ok(XmlNode::Text(text.trim().to_string()))
    }
}

fn unescape_xml(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_element() {
        let mut parser = XmlParser::new("<root>hello</root>");
        let node = parser.parse().unwrap();
        assert_eq!(node.tag_name(), Some("root"));
        assert_eq!(node.text_content(), "hello");
    }

    #[test]
    fn test_parse_attributes() {
        let mut parser = XmlParser::new(r#"<div class="foo" id="bar">text</div>"#);
        let node = parser.parse().unwrap();
        assert_eq!(node.attribute("class"), Some("foo"));
        assert_eq!(node.attribute("id"), Some("bar"));
    }

    #[test]
    fn test_parse_self_closing() {
        let mut parser = XmlParser::new("<br />");
        let node = parser.parse().unwrap();
        assert_eq!(node.tag_name(), Some("br"));
        assert!(node.children().is_empty());
    }

    #[test]
    fn test_parse_nested() {
        let xml = "<root><child1>a</child1><child2>b</child2></root>";
        let mut parser = XmlParser::new(xml);
        let node = parser.parse().unwrap();
        assert_eq!(node.children().len(), 2);
    }

    #[test]
    fn test_parse_comment() {
        let xml = "<root><!-- comment --></root>";
        let mut parser = XmlParser::new(xml);
        let node = parser.parse().unwrap();
        assert_eq!(node.children().len(), 1);
    }

    #[test]
    fn test_parse_cdata() {
        let xml = "<root><![CDATA[<script>alert(1)</script>]]></root>";
        let mut parser = XmlParser::new(xml);
        let node = parser.parse().unwrap();
        assert_eq!(node.children().len(), 1);
    }

    #[test]
    fn test_parse_prolog() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?><root/>"#;
        let mut parser = XmlParser::new(xml);
        let node = parser.parse().unwrap();
        assert_eq!(node.tag_name(), Some("root"));
    }
}
