use omega_lang::stdlib::xml::parser::XmlParser;
use omega_lang::stdlib::xml::dom::XmlNode;

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
fn test_find_by_attribute() {
    let mut root = XmlNode::element("root");
    let mut item = XmlNode::element("item");
    item.set_attribute("id", "42");
    root.add_child(item);

    let found = root.find_by_attribute("id", "42");
    assert_eq!(found.len(), 1);
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
fn test_comment() {
    let xml = "<root><!-- comment --></root>";
    let mut parser = XmlParser::new(xml);
    let node = parser.parse().unwrap();
    assert_eq!(node.children().len(), 1);
}

#[test]
fn test_cdata() {
    let xml = "<root><![CDATA[<script>alert(1)</script>]]></root>";
    let mut parser = XmlParser::new(xml);
    let node = parser.parse().unwrap();
    assert_eq!(node.children().len(), 1);
}

#[test]
fn test_prolog() {
    let xml = r#"<?xml version="1.0" encoding="UTF-8"?><root/>"#;
    let mut parser = XmlParser::new(xml);
    let node = parser.parse().unwrap();
    assert_eq!(node.tag_name(), Some("root"));
}
