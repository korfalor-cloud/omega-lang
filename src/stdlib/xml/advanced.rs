/// Advanced XML processing: SAX parser, DOM builder, XPath evaluator,
/// schema validation, and XSLT transformation.

use super::dom::XmlNode;
use super::parser::XmlParser;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// SAX Parser -- event-driven streaming parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum SaxEvent {
    StartElement {
        tag: String,
        attributes: HashMap<String, String>,
    },
    EndElement {
        tag: String,
    },
    Characters(String),
    CData(String),
    Comment(String),
}

pub struct SaxParser {
    input: Vec<char>,
    pos: usize,
}

impl SaxParser {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Vec<SaxEvent>, String> {
        let mut events = Vec::new();
        self.skip_prolog();
        self.parse_node(&mut events)?;
        Ok(events)
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

    fn skip_prolog(&mut self) {
        if self.pos + 1 < self.input.len()
            && self.input[self.pos] == '<'
            && self.input[self.pos + 1] == '?'
        {
            self.advance();
            self.advance();
            while let Some(c) = self.advance() {
                if c == '?' && self.peek() == Some('>') {
                    self.advance();
                    break;
                }
            }
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

    fn parse_node(&mut self, events: &mut Vec<SaxEvent>) -> Result<(), String> {
        loop {
            self.skip_whitespace();
            if self.pos >= self.input.len() {
                return Ok(());
            }
            if self.peek() != Some('<') {
                let text = self.read_until('<');
                let trimmed = text.trim().to_string();
                if !trimmed.is_empty() {
                    events.push(SaxEvent::Characters(trimmed));
                }
                continue;
            }

            // Detect comment
            if self.pos + 3 < self.input.len()
                && self.input[self.pos + 1] == '!'
                && self.input[self.pos + 2] == '-'
                && self.input[self.pos + 3] == '-'
            {
                for _ in 0..4 { self.advance(); }
                let mut comment = String::new();
                loop {
                    match self.advance() {
                        Some('-') if self.peek() == Some('-') => {
                            self.advance();
                            if self.peek() == Some('>') { self.advance(); }
                            break;
                        }
                        Some(c) => comment.push(c),
                        None => return Err("Unterminated comment".into()),
                    }
                }
                events.push(SaxEvent::Comment(comment));
                continue;
            }

            // Detect closing tag
            if self.pos + 1 < self.input.len() && self.input[self.pos + 1] == '/' {
                self.advance(); // <
                self.advance(); // /
                let tag = self.read_tag_name();
                self.skip_whitespace();
                if self.peek() == Some('>') { self.advance(); }
                events.push(SaxEvent::EndElement { tag });
                return Ok(());
            }

            // Opening tag
            self.advance(); // <
            let tag = self.read_tag_name();
            self.skip_whitespace();
            let mut attrs = HashMap::new();
            while let Some(c) = self.peek() {
                if c == '>' || c == '/' { break; }
                let (k, v) = self.read_attribute()?;
                attrs.insert(k, v);
                self.skip_whitespace();
            }

            events.push(SaxEvent::StartElement {
                tag: tag.clone(),
                attributes: attrs,
            });

            // Self-closing
            if self.peek() == Some('/') {
                self.advance();
                if self.peek() == Some('>') { self.advance(); }
                events.push(SaxEvent::EndElement { tag });
                continue;
            }

            if self.peek() == Some('>') { self.advance(); }

            // Parse children recursively
            self.parse_node(events)?;
        }
    }

    fn read_until(&mut self, stop: char) -> String {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c == stop { break; }
            s.push(c);
            self.advance();
        }
        s
    }

    fn read_tag_name(&mut self) -> String {
        let mut name = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' || c == '-' || c == ':' || c == '.' {
                name.push(c);
                self.advance();
            } else {
                break;
            }
        }
        name
    }

    fn read_attribute(&mut self) -> Result<(String, String), String> {
        let key = self.read_tag_name();
        self.skip_whitespace();
        if self.peek() != Some('=') { return Err("Expected '='".into()); }
        self.advance();
        self.skip_whitespace();
        let quote = self.advance().ok_or("Expected quote")?;
        if quote != '"' && quote != '\'' { return Err("Invalid quote".into()); }
        let mut value = String::new();
        loop {
            match self.advance() {
                Some(c) if c == quote => break,
                Some(c) => value.push(c),
                None => return Err("Unterminated attribute".into()),
            }
        }
        Ok((key, value))
    }
}

// ---------------------------------------------------------------------------
// DOM Builder -- programmatic tree construction
// ---------------------------------------------------------------------------

pub struct DomBuilder {
    stack: Vec<XmlNode>,
}

impl DomBuilder {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    pub fn start_element(&mut self, tag: &str, attrs: HashMap<String, String>) {
        let mut node = XmlNode::element(tag);
        for (k, v) in attrs {
            node.set_attribute(&k, &v);
        }
        self.stack.push(node);
    }

    pub fn end_element(&mut self) -> Option<XmlNode> {
        if self.stack.len() <= 1 {
            return self.stack.pop();
        }
        let child = self.stack.pop()?;
        self.stack.last_mut()?.add_child(child);
        None
    }

    pub fn add_text(&mut self, text: &str) {
        if let Some(top) = self.stack.last_mut() {
            top.add_child(XmlNode::text(text));
        }
    }

    pub fn add_comment(&mut self, text: &str) {
        if let Some(top) = self.stack.last_mut() {
            top.add_child(XmlNode::comment(text));
        }
    }

    pub fn build(mut self) -> Option<XmlNode> {
        while self.stack.len() > 1 {
            self.end_element();
        }
        self.stack.pop()
    }
}

// ---------------------------------------------------------------------------
// XPath Evaluator -- subset supporting /, //, tag, [@attr=val], [n]
// ---------------------------------------------------------------------------

pub fn xpath_select<'a>(node: &'a XmlNode, expr: &str) -> Vec<&'a XmlNode> {
    let steps: Vec<&str> = expr.split('/').filter(|s| !s.is_empty()).collect();
    let mut current: Vec<&XmlNode> = vec![node];
    let mut starts_at_root = expr.starts_with('/');

    for step in &steps {
        let is_desc = starts_at_root && step == &steps[0] && expr.starts_with("//");
        starts_at_root = false;

        let (tag, predicate) = parse_step(step);
        let mut next = Vec::new();

        for n in &current {
            let candidates: Vec<&XmlNode> = if is_desc || step == &"" {
                collect_descendants(n)
            } else {
                n.children().iter().collect()
            };

            for c in candidates {
                if tag.is_empty() || c.tag_name() == Some(tag.as_str()) {
                    next.push(c);
                }
            }
        }

        // Apply predicate
        if let Some(pred) = predicate {
            if pred.starts_with('@') {
                // [@attr=val]
                let inner = &pred[1..];
                if let Some((ak, av)) = inner.split_once('=') {
                    next.retain(|n| n.attribute(ak.trim()) == Some(av.trim()));
                }
            } else if let Ok(idx) = pred.parse::<usize>() {
                // [n] -- 1-based
                if idx >= 1 && idx <= next.len() {
                    next = vec![next[idx - 1]];
                } else {
                    next.clear();
                }
            }
        }

        current = next;
    }
    current
}

fn parse_step(step: &str) -> (String, Option<String>) {
    if let Some(bracket_pos) = step.find('[') {
        let tag = step[..bracket_pos].to_string();
        let pred = step[bracket_pos + 1..].trim_end_matches(']').to_string();
        (tag, Some(pred))
    } else {
        (step.to_string(), None)
    }
}

fn collect_descendants<'a>(node: &'a XmlNode) -> Vec<&'a XmlNode> {
    let mut result = Vec::new();
    for child in node.children() {
        result.push(child);
        result.extend(collect_descendants(child));
    }
    result
}

// ---------------------------------------------------------------------------
// Schema Validation -- basic structural checks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SchemaRule {
    pub tag: String,
    pub required_children: Vec<String>,
    pub required_attrs: Vec<String>,
    pub allowed_children: Option<Vec<String>>,
}

pub struct XmlSchema {
    pub rules: HashMap<String, SchemaRule>,
}

impl XmlSchema {
    pub fn new() -> Self {
        Self { rules: HashMap::new() }
    }

    pub fn add_rule(&mut self, rule: SchemaRule) {
        self.rules.insert(rule.tag.clone(), rule);
    }

    pub fn validate(&self, node: &XmlNode) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        self.validate_node(node, &mut errors);
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    fn validate_node(&self, node: &XmlNode, errors: &mut Vec<String>) {
        if let XmlNode::Element { tag, attributes, children } = node {
            if let Some(rule) = self.rules.get(tag.as_str()) {
                for attr in &rule.required_attrs {
                    if !attributes.contains_key(attr.as_str()) {
                        errors.push(format!("<{}>: missing required attribute '{}'", tag, attr));
                    }
                }
                let child_tags: Vec<&str> = children
                    .iter()
                    .filter_map(|c| c.tag_name())
                    .collect();
                for req in &rule.required_children {
                    if !child_tags.contains(&req.as_str()) {
                        errors.push(format!("<{}>: missing required child '<{}>'", tag, req));
                    }
                }
                if let Some(ref allowed) = rule.allowed_children {
                    for ct in &child_tags {
                        if !allowed.contains(&ct.to_string()) {
                            errors.push(format!("<{}>: unexpected child '<{}>'", tag, ct));
                        }
                    }
                }
            }
            for child in children {
                self.validate_node(child, errors);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// XSLT Transformation -- simplified template-based transform
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct XsltTemplate {
    pub match_tag: String,
    pub output_tag: Option<String>,
    pub copy_attrs: bool,
    pub text_override: Option<String>,
    pub children_of: Option<String>,
}

pub struct XsltStylesheet {
    pub templates: Vec<XsltTemplate>,
    pub default_copy: bool,
}

impl XsltStylesheet {
    pub fn new() -> Self {
        Self { templates: Vec::new(), default_copy: true }
    }

    pub fn add_template(&mut self, t: XsltTemplate) {
        self.templates.push(t);
    }

    pub fn transform(&self, node: &XmlNode) -> XmlNode {
        self.apply(node)
    }

    fn apply(&self, node: &XmlNode) -> XmlNode {
        if let XmlNode::Element { tag, attributes, children } = node {
            // Find matching template
            for tmpl in &self.templates {
                if tmpl.match_tag == *tag {
                    let out_tag = tmpl.output_tag.as_deref().unwrap_or(tag);
                    let mut result = XmlNode::element(out_tag);

                    if tmpl.copy_attrs {
                        for (k, v) in attributes {
                            result.set_attribute(k, v);
                        }
                    }

                    if let Some(ref text) = tmpl.text_override {
                        result.add_child(XmlNode::text(text));
                    } else if let Some(ref source_tag) = tmpl.children_of {
                        // Copy children from matched source element
                        for child in children {
                            if child.tag_name() == Some(source_tag.as_str()) {
                                for gc in child.children() {
                                    result.add_child(self.apply(gc));
                                }
                            }
                        }
                    } else {
                        for child in children {
                            result.add_child(self.apply(child));
                        }
                    }

                    return result;
                }
            }

            // Default: identity transform
            if self.default_copy {
                let mut result = XmlNode::element(tag);
                for (k, v) in attributes {
                    result.set_attribute(k, v);
                }
                for child in children {
                    result.add_child(self.apply(child));
                }
                return result;
            }

            XmlNode::element(tag)
        } else {
            node.clone()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SAX parser tests --

    #[test]
    fn test_sax_simple() {
        let xml = "<root><child>text</child></root>";
        let events = SaxParser::new(xml).parse().unwrap();
        assert!(matches!(&events[0], SaxEvent::StartElement { tag, .. } if tag == "root"));
        assert!(matches!(&events[1], SaxEvent::StartElement { tag, .. } if tag == "child"));
        assert!(matches!(&events[2], SaxEvent::Characters(s) if s == "text"));
        assert!(matches!(&events[3], SaxEvent::EndElement { tag } if tag == "child"));
        assert!(matches!(&events[4], SaxEvent::EndElement { tag } if tag == "root"));
    }

    #[test]
    fn test_sax_attributes() {
        let xml = r#"<item id="1" name="foo"/>"#;
        let events = SaxParser::new(xml).parse().unwrap();
        if let SaxEvent::StartElement { attributes, .. } = &events[0] {
            assert_eq!(attributes.get("id").map(|s| s.as_str()), Some("1"));
            assert_eq!(attributes.get("name").map(|s| s.as_str()), Some("foo"));
        } else {
            panic!("Expected StartElement");
        }
    }

    #[test]
    fn test_sax_comment() {
        let xml = "<root><!-- hello --></root>";
        let events = SaxParser::new(xml).parse().unwrap();
        assert!(matches!(&events[1], SaxEvent::Comment(s) if s == " hello "));
    }

    // -- DOM builder tests --

    #[test]
    fn test_dom_builder() {
        let mut b = DomBuilder::new();
        b.start_element("root", HashMap::new());
        b.start_element("child", HashMap::new());
        b.add_text("hello");
        b.end_element();
        let root = b.build().unwrap();
        assert_eq!(root.tag_name(), Some("root"));
        assert_eq!(root.text_content(), "hello");
    }

    // -- XPath tests --

    #[test]
    fn test_xpath_child() {
        let mut root = XmlNode::element("root");
        let mut child = XmlNode::element("item");
        child.add_child(XmlNode::text("val"));
        root.add_child(child);

        let res = xpath_select(&root, "item");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].text_content(), "val");
    }

    #[test]
    fn test_xpath_descendant() {
        let mut root = XmlNode::element("root");
        let mut mid = XmlNode::element("mid");
        let mut deep = XmlNode::element("target");
        deep.add_child(XmlNode::text("found"));
        mid.add_child(deep);
        root.add_child(mid);

        let res = xpath_select(&root, "//target");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].text_content(), "found");
    }

    #[test]
    fn test_xpath_attribute_predicate() {
        let mut root = XmlNode::element("root");
        let mut a = XmlNode::element("item");
        a.set_attribute("id", "1");
        let mut b = XmlNode::element("item");
        b.set_attribute("id", "2");
        root.add_child(a);
        root.add_child(b);

        let res = xpath_select(&root, "item[@id=2]");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].attribute("id"), Some("2"));
    }

    #[test]
    fn test_xpath_position_predicate() {
        let mut root = XmlNode::element("root");
        for i in 1..=3 {
            let mut n = XmlNode::element("item");
            n.add_child(XmlNode::text(&i.to_string()));
            root.add_child(n);
        }
        let res = xpath_select(&root, "item[2]");
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].text_content(), "2");
    }

    // -- Schema validation tests --

    #[test]
    fn test_schema_valid() {
        let schema = {
            let mut s = XmlSchema::new();
            s.add_rule(SchemaRule {
                tag: "root".into(),
                required_children: vec!["child".into()],
                required_attrs: vec!["id".into()],
                allowed_children: None,
            });
            s
        };
        let mut root = XmlNode::element("root");
        root.set_attribute("id", "1");
        root.add_child(XmlNode::element("child"));
        assert!(schema.validate(&root).is_ok());
    }

    #[test]
    fn test_schema_missing_child() {
        let schema = {
            let mut s = XmlSchema::new();
            s.add_rule(SchemaRule {
                tag: "root".into(),
                required_children: vec!["child".into()],
                required_attrs: vec![],
                allowed_children: None,
            });
            s
        };
        let root = XmlNode::element("root");
        let err = schema.validate(&root).unwrap_err();
        assert!(err[0].contains("missing required child"));
    }

    #[test]
    fn test_schema_missing_attr() {
        let schema = {
            let mut s = XmlSchema::new();
            s.add_rule(SchemaRule {
                tag: "root".into(),
                required_children: vec![],
                required_attrs: vec!["id".into()],
                allowed_children: None,
            });
            s
        };
        let root = XmlNode::element("root");
        let err = schema.validate(&root).unwrap_err();
        assert!(err[0].contains("missing required attribute"));
    }

    #[test]
    fn test_schema_unexpected_child() {
        let schema = {
            let mut s = XmlSchema::new();
            s.add_rule(SchemaRule {
                tag: "root".into(),
                required_children: vec![],
                required_attrs: vec![],
                allowed_children: Some(vec!["allowed".into()]),
            });
            s
        };
        let mut root = XmlNode::element("root");
        root.add_child(XmlNode::element("forbidden"));
        let err = schema.validate(&root).unwrap_err();
        assert!(err[0].contains("unexpected child"));
    }

    // -- XSLT transformation tests --

    #[test]
    fn test_xslt_identity() {
        let mut root = XmlNode::element("root");
        root.add_child(XmlNode::element("child"));

        let sheet = XsltStylesheet::new();
        let result = sheet.transform(&root);
        assert_eq!(result.tag_name(), Some("root"));
        assert_eq!(result.children().len(), 1);
    }

    #[test]
    fn test_xslt_rename() {
        let mut root = XmlNode::element("root");
        let mut item = XmlNode::element("item");
        item.add_child(XmlNode::text("val"));
        root.add_child(item);

        let mut sheet = XsltStylesheet::new();
        sheet.add_template(XsltTemplate {
            match_tag: "item".into(),
            output_tag: Some("entry".into()),
            copy_attrs: false,
            text_override: None,
            children_of: None,
        });
        let result = sheet.transform(&root);
        assert_eq!(result.children()[0].tag_name(), Some("entry"));
        assert_eq!(result.children()[0].text_content(), "val");
    }

    #[test]
    fn test_xslt_text_override() {
        let mut root = XmlNode::element("root");
        root.add_child(XmlNode::element("secret"));

        let mut sheet = XsltStylesheet::new();
        sheet.add_template(XsltTemplate {
            match_tag: "secret".into(),
            output_tag: None,
            copy_attrs: false,
            text_override: Some("REDACTED".into()),
            children_of: None,
        });
        let result = sheet.transform(&root);
        assert_eq!(result.children()[0].text_content(), "REDACTED");
    }
}
