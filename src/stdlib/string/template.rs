use std::collections::HashMap;

pub struct TemplateEngine {
    delimiters: (String, String),
    escape_html: bool,
}

impl TemplateEngine {
    pub fn new() -> Self {
        Self {
            delimiters: ("{{".to_string(), "}}".to_string()),
            escape_html: true,
        }
    }

    pub fn with_delimiters(open: &str, close: &str) -> Self {
        Self {
            delimiters: (open.to_string(), close.to_string()),
            escape_html: true,
        }
    }

    pub fn with_escape_html(mut self, escape: bool) -> Self {
        self.escape_html = escape;
        self
    }

    pub fn render(&self, template: &str, context: &HashMap<String, String>) -> String {
        let mut result = String::new();
        let (open, close) = &self.delimiters;
        let mut remaining = template;

        while let Some(start) = remaining.find(open.as_str()) {
            result.push_str(&remaining[..start]);
            remaining = &remaining[start + open.len()..];

            if let Some(end) = remaining.find(close.as_str()) {
                let key = remaining[..end].trim();
                remaining = &remaining[end + close.len()..];

                if let Some(value) = context.get(key) {
                    if self.escape_html {
                        result.push_str(&self.escape(value));
                    } else {
                        result.push_str(value);
                    }
                }
            } else {
                result.push_str(open);
            }
        }

        result.push_str(remaining);
        result
    }

    pub fn render_with_blocks(
        &self,
        template: &str,
        context: &HashMap<String, String>,
        blocks: &HashMap<String, String>,
    ) -> String {
        let mut result = template.to_string();

        // Replace block placeholders
        for (name, content) in blocks {
            let placeholder = format!("{{{{ block '{}' }}}}", name);
            result = result.replace(&placeholder, content);
        }

        // Replace variable placeholders
        self.render(&result, context)
    }

    fn escape(&self, s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
    }

    pub fn unescape(s: &str) -> String {
        s.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#x27;", "'")
    }
}

// String interpolation
pub fn interpolate(template: &str, args: &[&str]) -> String {
    let mut result = String::new();
    let mut arg_index = 0;
    let mut chars = template.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '{' {
            if chars.peek() == Some(&'}') {
                chars.next();
                if arg_index < args.len() {
                    result.push_str(args[arg_index]);
                    arg_index += 1;
                }
            } else {
                // Parse format spec
                let mut spec = String::new();
                while let Some(&c) = chars.peek() {
                    if c == '}' {
                        chars.next();
                        break;
                    }
                    spec.push(c);
                    chars.next();
                }

                if arg_index < args.len() {
                    result.push_str(args[arg_index]);
                    arg_index += 1;
                }
            }
        } else if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('{') => result.push('{'),
                Some('}') => result.push('}'),
                Some(c) => {
                    result.push('\\');
                    result.push(c);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }

    result
}

// Mustache-like template
pub struct MustacheTemplate {
    template: String,
}

impl MustacheTemplate {
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
        }
    }

    pub fn render(&self, context: &HashMap<String, serde_json::Value>) -> String {
        let mut result = self.template.clone();

        // Replace {{variable}} with values
        for (key, value) in context {
            let placeholder = format!("{{{{{}}}}}", key);
            let replacement = match value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Null => String::new(),
                _ => value.to_string(),
            };
            result = result.replace(&placeholder, &replacement);
        }

        result
    }

    pub fn render_section(
        &self,
        context: &HashMap<String, serde_json::Value>,
    ) -> String {
        let mut result = self.template.clone();

        // Handle sections {{#section}}...{{/section}}
        let mut i = 0;
        while i < result.len() {
            if result[i..].starts_with("{{#") {
                if let Some(end) = result[i..].find("}}") {
                    let section_name = result[i + 3..i + end].trim();
                    let section_start = i + end + 2;

                    if let Some(section_end) = result[section_start..].find(&format!("{{/{}}}", section_name)) {
                        let section_content = &result[section_start..section_start + section_end];

                        if let Some(value) = context.get(section_name) {
                            let rendered = match value {
                                serde_json::Value::Array(arr) => {
                                    let mut rendered = String::new();
                                    for item in arr {
                                        let mut item_context = context.clone();
                                        if let serde_json::Value::Object(obj) = item {
                                            for (k, v) in obj {
                                                item_context.insert(k.clone(), v.clone());
                                            }
                                        }
                                        let template = MustacheTemplate::new(section_content);
                                        rendered.push_str(&template.render(&item_context));
                                    }
                                    rendered
                                }
                                serde_json::Value::Bool(true) => {
                                    let template = MustacheTemplate::new(section_content);
                                    template.render(context)
                                }
                                _ => String::new(),
                            };

                            result = format!(
                                "{}{}{}",
                                &result[..i],
                                rendered,
                                &result[section_start + section_end + section_name.len() + 4..]
                            );
                        }
                    }
                }
            }
            i += 1;
        }

        result
    }
}

// Format string utilities
pub fn format_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();

    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.len());
            }
        }
    }

    let mut result = String::new();

    // Header
    for (i, header) in headers.iter().enumerate() {
        if i > 0 {
            result.push_str(" | ");
        }
        result.push_str(&format!("{:width$}", header, width = widths[i]));
    }
    result.push('\n');

    // Separator
    for (i, width) in widths.iter().enumerate() {
        if i > 0 {
            result.push_str("-+-");
        }
        result.push_str(&"-".repeat(*width));
    }
    result.push('\n');

    // Rows
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i > 0 {
                result.push_str(" | ");
            }
            result.push_str(&format!("{:width$}", cell, width = widths[i]));
        }
        result.push('\n');
    }

    result
}

pub fn format_list(items: &[&str]) -> String {
    items
        .iter()
        .enumerate()
        .map(|(i, item)| format!("{}. {}", i + 1, item))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn format_tree(root: &str, children: &[(&str, &[&str])]) -> String {
    let mut result = format!("{}\n", root);

    for (i, (parent, items)) in children.iter().enumerate() {
        let is_last_parent = i == children.len() - 1;
        let prefix = if is_last_parent { "└── " } else { "├── " };
        result.push_str(&format!("{}{}\n", prefix, parent));

        for (j, item) in items.iter().enumerate() {
            let is_last = j == items.len() - 1;
            let child_prefix = if is_last_parent {
                "    "
            } else {
                "│   "
            };
            let item_prefix = if is_last { "└── " } else { "├── " };
            result.push_str(&format!("{}{}{}\n", child_prefix, item_prefix, item));
        }
    }

    result
}
