/// XML DOM representation.

use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub enum XmlNode {
    Element {
        tag: String,
        attributes: HashMap<String, String>,
        children: Vec<XmlNode>,
    },
    Text(String),
    CData(String),
    Comment(String),
    ProcessingInstruction {
        target: String,
        data: String,
    },
}

impl XmlNode {
    pub fn element(tag: &str) -> Self {
        XmlNode::Element {
            tag: tag.to_string(),
            attributes: HashMap::new(),
            children: Vec::new(),
        }
    }

    pub fn text(content: &str) -> Self {
        XmlNode::Text(content.to_string())
    }

    pub fn comment(content: &str) -> Self {
        XmlNode::Comment(content.to_string())
    }

    pub fn is_element(&self) -> bool {
        matches!(self, XmlNode::Element { .. })
    }

    pub fn is_text(&self) -> bool {
        matches!(self, XmlNode::Text(_))
    }

    pub fn tag_name(&self) -> Option<&str> {
        match self {
            XmlNode::Element { tag, .. } => Some(tag),
            _ => None,
        }
    }

    pub fn attributes(&self) -> Option<&HashMap<String, String>> {
        match self {
            XmlNode::Element { attributes, .. } => Some(attributes),
            _ => None,
        }
    }

    pub fn attribute(&self, name: &str) -> Option<&str> {
        self.attributes()?.get(name).map(|s| s.as_str())
    }

    pub fn children(&self) -> &[XmlNode] {
        match self {
            XmlNode::Element { children, .. } => children,
            _ => &[],
        }
    }

    pub fn children_mut(&mut self) -> Option<&mut Vec<XmlNode>> {
        match self {
            XmlNode::Element { children, .. } => Some(children),
            _ => None,
        }
    }

    pub fn add_child(&mut self, child: XmlNode) {
        if let Some(children) = self.children_mut() {
            children.push(child);
        }
    }

    pub fn set_attribute(&mut self, key: &str, value: &str) {
        if let XmlNode::Element { attributes, .. } = self {
            attributes.insert(key.to_string(), value.to_string());
        }
    }

    pub fn text_content(&self) -> String {
        match self {
            XmlNode::Text(s) => s.clone(),
            XmlNode::CData(s) => s.clone(),
            XmlNode::Element { children, .. } => {
                children.iter().map(|c| c.text_content()).collect::<Vec<_>>().join("")
            }
            _ => String::new(),
        }
    }

    pub fn find_elements(&self, tag: &str) -> Vec<&XmlNode> {
        let mut result = Vec::new();
        self.find_elements_recursive(tag, &mut result);
        result
    }

    fn find_elements_recursive<'a>(&'a self, tag: &str, result: &mut Vec<&'a XmlNode>) {
        if let XmlNode::Element { tag: t, children, .. } = self {
            if t == tag {
                result.push(self);
            }
            for child in children {
                child.find_elements_recursive(tag, result);
            }
        }
    }

    pub fn find_element(&self, tag: &str) -> Option<&XmlNode> {
        self.find_elements(tag).into_iter().next()
    }

    pub fn find_by_attribute(&self, attr_name: &str, attr_value: &str) -> Vec<&XmlNode> {
        let mut result = Vec::new();
        self.find_by_attribute_recursive(attr_name, attr_value, &mut result);
        result
    }

    fn find_by_attribute_recursive<'a>(&'a self, attr_name: &str, attr_value: &str, result: &mut Vec<&'a XmlNode>) {
        if let XmlNode::Element { attributes, children, .. } = self {
            if attributes.get(attr_name).map(|s| s.as_str()) == Some(attr_value) {
                result.push(self);
            }
            for child in children {
                child.find_by_attribute_recursive(attr_name, attr_value, result);
            }
        }
    }

    pub fn to_xml(&self) -> String {
        let mut output = String::new();
        self.to_xml_impl(&mut output, 0);
        output
    }

    pub fn to_xml_pretty(&self) -> String {
        let mut output = String::new();
        self.to_xml_impl(&mut output, 0);
        output
    }

    fn to_xml_impl(&self, output: &mut String, indent: usize) {
        let pad = "  ".repeat(indent);
        match self {
            XmlNode::Element { tag, attributes, children } => {
                output.push_str(&format!("{}<{}", pad, tag));
                for (key, value) in attributes {
                    output.push_str(&format!(" {}=\"{}\"", key, escape_xml(value)));
                }
                if children.is_empty() {
                    output.push_str(" />");
                } else {
                    output.push('>');
                    let all_text = children.iter().all(|c| matches!(c, XmlNode::Text(_)));
                    if all_text {
                        for child in children {
                            child.to_xml_impl(output, 0);
                        }
                    } else {
                        output.push('\n');
                        for child in children {
                            child.to_xml_impl(output, indent + 1);
                            output.push('\n');
                        }
                        output.push_str(&format!("{}</{}", pad, tag));
                    }
                    output.push('>');
                }
            }
            XmlNode::Text(s) => {
                output.push_str(&escape_xml(s));
            }
            XmlNode::CData(s) => {
                output.push_str(&format!("<![CDATA[{}]]>", s));
            }
            XmlNode::Comment(s) => {
                output.push_str(&format!("<!--{}-->", s));
            }
            XmlNode::ProcessingInstruction { target, data } => {
                output.push_str(&format!("<?{} {}?>", target, data));
            }
        }
    }
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

impl fmt::Display for XmlNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_xml())
    }
}

/// XML document wrapper
#[derive(Debug, Clone)]
pub struct XmlDocument {
    pub prolog: Option<String>,
    pub root: XmlNode,
}

impl XmlDocument {
    pub fn new(root: XmlNode) -> Self {
        Self { prolog: None, root }
    }

    pub fn with_prolog(mut self, version: &str, encoding: &str) -> Self {
        self.prolog = Some(format!("<?xml version=\"{}\" encoding=\"{}\"?>", version, encoding));
        self
    }

    pub fn to_xml(&self) -> String {
        let mut output = String::new();
        if let Some(prolog) = &self.prolog {
            output.push_str(prolog);
            output.push('\n');
        }
        output.push_str(&self.root.to_xml());
        output
    }
}

impl fmt::Display for XmlDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_xml())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_creation() {
        let mut elem = XmlNode::element("div");
        elem.set_attribute("class", "container");
        elem.add_child XmlNode::text("Hello"));

        assert_eq!(elem.tag_name(), Some("div"));
        assert_eq!(elem.attribute("class"), Some("container"));
        assert_eq!(elem.text_content(), "Hello");
    }

    #[test]
    fn test_find_elements() {
        let mut root = XmlNode::element("root");
        let mut child1 = XmlNode::element("item");
        child1.add_child(XmlNode::text("one"));
        let mut child2 = XmlNode::element("item");
        child2.add_child(XmlNode::text("two"));
        root.add_child(child1);
        root.add_child(child2);

        let items = root.find_elements("item");
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_to_xml() {
        let mut root = XmlNode::element("root");
        let mut child = XmlNode::element("child");
        child.add_child(XmlNode::text("content"));
        root.add_child(child);

        let xml = root.to_xml();
        assert!(xml.contains("<root>"));
        assert!(xml.contains("<child>"));
        assert!(xml.contains("content"));
    }

    #[test]
    fn test_escape_xml() {
        let mut elem = XmlNode::element("test");
        elem.add_child(XmlNode::text("<script>alert('xss')</script>"));
        let xml = elem.to_xml();
        assert!(xml.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_find_by_attribute() {
        let mut root = XmlNode::element("root");
        let mut item = XmlNode::element("item");
        item.set_attribute("id", "42");
        root.add_child(item);

        let found = root.find_by_attribute("id", "42");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_document() {
        let root = XmlNode::element("root");
        let doc = XmlDocument::new(root).with_prolog("1.0", "UTF-8");
        let xml = doc.to_xml();
        assert!(xml.contains("<?xml"));
    }
}
