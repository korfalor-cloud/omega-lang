/// Web template engine for HTML generation.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct WebTemplate {
    template: String,
    variables: HashMap<String, String>,
    blocks: HashMap<String, String>,
    parent: Option<Box<WebTemplate>>,
}

impl WebTemplate {
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
            variables: HashMap::new(),
            blocks: HashMap::new(),
            parent: None,
        }
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.variables.insert(key.to_string(), value.to_string());
    }

    pub fn set_bool(&mut self, key: &str, value: bool) {
        self.variables.insert(key.to_string(), if value { "true".to_string() } else { "false".to_string() });
    }

    pub fn set_number(&mut self, key: &str, value: f64) {
        self.variables.insert(key.to_string(), value.to_string());
    }

    pub fn set_block(&mut self, name: &str, content: &str) {
        self.blocks.insert(name.to_string(), content.to_string());
    }

    pub fn extend(parent: WebTemplate) -> Self {
        Self {
            template: String::new(),
            variables: HashMap::new(),
            blocks: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    pub fn render(&self) -> String {
        let mut output = if let Some(parent) = &self.parent {
            parent.render()
        } else {
            self.template.clone()
        };

        // Replace variables: {{var}}
        for (key, value) in &self.variables {
            let pattern = format!("{{{{{}}}}}", key);
            output = output.replace(&pattern, value);
        }

        // Replace blocks: {% block name %}...{% endblock %}
        for (name, content) in &self.blocks {
            let start = format!("{{% block {} %}}", name);
            let end = "{% endblock %}";
            if let Some(start_pos) = output.find(&start) {
                if let Some(end_pos) = output[start_pos..].find(end) {
                    let end_pos = start_pos + end_pos + end.len();
                    output = format!("{}{}{}", &output[..start_pos], content, &output[end_pos..]);
                }
            }
        }

        // Handle conditionals: {% if var %}...{% endif %}
        output = self.process_conditionals(&output);

        // Handle loops: {% for item in list %}...{% endfor %}
        output = self.process_loops(&output);

        output
    }

    fn process_conditionals(&self, template: &str) -> String {
        let mut output = template.to_string();
        let mut i = 0;
        let chars: Vec<char> = output.chars().collect();
        let mut result = String::new();

        while i < chars.len() {
            if i + 4 < chars.len()
                && chars[i] == '{' && chars[i + 1] == '%'
                && chars[i + 2] == ' '
            {
                let rest: String = chars[i..].iter().collect();
                if rest.starts_with("{% if ") {
                    let var_end = rest.find(" %}").unwrap_or(rest.len());
                    let var_name = rest[6..var_end].trim();
                    let endif_marker = "{% endif %}";
                    if let Some(endif_pos) = rest.find(endif_marker) {
                        let inner = &rest[var_end + 3..endif_pos];
                        let value = self.variables.get(var_name).map(|s| s.as_str()).unwrap_or("");
                        if !value.is_empty() && value != "false" && value != "0" {
                            result.push_str(inner);
                        }
                        i += var_end + 3 + inner.len() + endif_marker.len();
                        continue;
                    }
                }
            }
            result.push(chars[i]);
            i += 1;
        }

        result
    }

    fn process_loops(&self, template: &str) -> String {
        let mut output = template.to_string();
        // Simple loop processing
        while let Some(start) = output.find("{% for ") {
            let end_marker = "{% endfor %}";
            if let Some(end) = output[start..].find(end_marker) {
                let end = start + end + end_marker.len();
                let block = &output[start..end];

                // Parse: {% for item in list %}...{% endfor %}
                let header_end = block.find(" %}").unwrap_or(block.len());
                let header = &block[..header_end];
                let body = &block[header_end + 3..block.len() - end_marker.len()];

                let parts: Vec<&str> = header.split_whitespace().collect();
                if parts.len() >= 4 && parts[2] == "in" {
                    let item_name = parts[1];
                    let list_name = parts[3];

                    if let Some(list_str) = self.variables.get(list_name) {
                        let items: Vec<&str> = list_str.split(',').collect();
                        let mut rendered = String::new();
                        for item in items {
                            rendered.push_str(&body.replace(
                                &format!("{{{{{}}}}}", item_name),
                                item.trim(),
                            ));
                        }
                        output = format!("{}{}{}", &output[..start], rendered, &output[end..]);
                    } else {
                        output = format!("{}{}{}", &output[..start], "", &output[end..]);
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        output
    }
}

/// HTML helper functions
pub struct Html;

impl Html {
    pub fn escape(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
    }

    pub fn tag(name: &str, attrs: &HashMap<&str, &str>, content: &str) -> String {
        let attr_str = attrs.iter()
            .map(|(k, v)| format!(" {}=\"{}\"", k, Self::escape(v)))
            .collect::<Vec<_>>()
            .join("");
        format!("<{}{}>{}</{}>", name, attr_str, Self::escape(content), name)
    }

    pub fn void_tag(name: &str, attrs: &HashMap<&str, &str>) -> String {
        let attr_str = attrs.iter()
            .map(|(k, v)| format!(" {}=\"{}\"", k, Self::escape(v)))
            .collect::<Vec<_>>()
            .join("");
        format!("<{}{} />", name, attr_str)
    }

    pub fn a(href: &str, text: &str) -> String {
        format!("<a href=\"{}\">{}</a>", Self::escape(href), Self::escape(text))
    }

    pub fn img(src: &str, alt: &str) -> String {
        format!("<img src=\"{}\" alt=\"{}\" />", Self::escape(src), Self::escape(alt))
    }

    pub fn input(name: &str, value: &str, input_type: &str) -> String {
        format!("<input type=\"{}\" name=\"{}\" value=\"{}\" />",
            Self::escape(input_type), Self::escape(name), Self::escape(value))
    }

    pub fn select(name: &str, options: &[(&str, &str)], selected: &str) -> String {
        let opts = options.iter()
            .map(|(val, label)| {
                let sel = if *val == selected { " selected" } else { "" };
                format!("<option value=\"{}\"{}>{}</option>", Self::escape(val), sel, Self::escape(label))
            })
            .collect::<Vec<_>>()
            .join("");
        format!("<select name=\"{}\">{}</select>", Self::escape(name), opts)
    }

    pub fn table(headers: &[&str], rows: &[Vec<&str>]) -> String {
        let mut html = String::from("<table><thead><tr>");
        for h in headers {
            html.push_str(&format!("<th>{}</th>", Self::escape(h)));
        }
        html.push_str("</tr></thead><tbody>");
        for row in rows {
            html.push_str("<tr>");
            for cell in row {
                html.push_str(&format!("<td>{}</td>", Self::escape(cell)));
            }
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table>");
        html
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_variables() {
        let mut tpl = WebTemplate::new("<h1>Hello, {{name}}!</h1>");
        tpl.set("name", "World");
        assert_eq!(tpl.render(), "<h1>Hello, World!</h1>");
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(Html::escape("<script>"), "&lt;script&gt;");
    }

    #[test]
    fn test_html_tag() {
        let html = Html::tag("p", &HashMap::new(), "Hello");
        assert_eq!(html, "<p>Hello</p>");
    }

    #[test]
    fn test_html_table() {
        let headers = vec!["Name", "Age"];
        let rows = vec![vec!["Alice", "30"], vec!["Bob", "25"]];
        let html = Html::table(&headers, &rows);
        assert!(html.contains("<th>Name</th>"));
        assert!(html.contains("<td>Alice</td>"));
    }

    #[test]
    fn test_html_link() {
        let html = Html::a("https://example.com", "Click");
        assert_eq!(html, "<a href=\"https://example.com\">Click</a>");
    }
}
