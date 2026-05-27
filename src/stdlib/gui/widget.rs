/// GUI widget system for terminal-based interfaces.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Rect {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub height: u16,
}

impl Rect {
    pub fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
        Self { x, y, width, height }
    }

    pub fn contains(&self, px: u16, py: u16) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    pub fn center(&self) -> (u16, u16) {
        (self.x + self.width / 2, self.y + self.height / 2)
    }
}

#[derive(Debug, Clone)]
pub enum Widget {
    Label {
        text: String,
        style: Style,
    },
    Button {
        text: String,
        style: Style,
        pressed: bool,
        on_click: Option<String>,
    },
    TextInput {
        value: String,
        placeholder: String,
        cursor: usize,
        style: Style,
    },
    CheckBox {
        text: String,
        checked: bool,
        style: Style,
    },
    Radio {
        text: String,
        selected: bool,
        group: String,
        style: Style,
    },
    ProgressBar {
        value: f64,
        max: f64,
        width: u16,
        style: Style,
    },
    List {
        items: Vec<String>,
        selected: usize,
        style: Style,
    },
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        selected_row: usize,
        style: Style,
    },
    Panel {
        title: String,
        children: Vec<Widget>,
        style: Style,
    },
    Container {
        children: Vec<Widget>,
    },
}

#[derive(Debug, Clone)]
pub struct Style {
    pub fg: Color,
    pub bg: Color,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub border: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Color {
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    Default,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            fg: Color::Default,
            bg: Color::Default,
            bold: false,
            italic: false,
            underline: false,
            border: false,
        }
    }
}

impl Widget {
    pub fn label(text: &str) -> Self {
        Widget::Label {
            text: text.to_string(),
            style: Style::default(),
        }
    }

    pub fn button(text: &str) -> Self {
        Widget::Button {
            text: text.to_string(),
            style: Style::default(),
            pressed: false,
            on_click: None,
        }
    }

    pub fn text_input(placeholder: &str) -> Self {
        Widget::TextInput {
            value: String::new(),
            placeholder: placeholder.to_string(),
            cursor: 0,
            style: Style::default(),
        }
    }

    pub fn checkbox(text: &str, checked: bool) -> Self {
        Widget::CheckBox {
            text: text.to_string(),
            checked,
            style: Style::default(),
        }
    }

    pub fn progress_bar(value: f64, max: f64) -> Self {
        Widget::ProgressBar {
            value,
            max,
            width: 20,
            style: Style::default(),
        }
    }

    pub fn list(items: Vec<String>) -> Self {
        Widget::List {
            items,
            selected: 0,
            style: Style::default(),
        }
    }

    pub fn table(headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        Widget::Table {
            headers,
            rows,
            selected_row: 0,
            style: Style::default(),
        }
    }

    pub fn panel(title: &str, children: Vec<Widget>) -> Self {
        Widget::Panel {
            title: title.to_string(),
            children,
            style: Style::default(),
        }
    }

    pub fn container(children: Vec<Widget>) -> Self {
        Widget::Container { children }
    }

    pub fn render(&self) -> String {
        match self {
            Widget::Label { text, .. } => text.clone(),
            Widget::Button { text, pressed, .. } => {
                if *pressed {
                    format!("[{}]", text)
                } else {
                    format!("<{}>", text)
                }
            }
            Widget::TextInput { value, placeholder, .. } => {
                if value.is_empty() {
                    format!("[{}]", placeholder)
                } else {
                    format!("[{}]", value)
                }
            }
            Widget::CheckBox { text, checked, .. } => {
                let mark = if *checked { "x" } else { " " };
                format!("[{}] {}", mark, text)
            }
            Widget::Radio { text, selected, .. } => {
                let mark = if *selected { "(*)" } else { "( )" };
                format!("{} {}", mark, text)
            }
            Widget::ProgressBar { value, max, width, .. } => {
                let filled = (value / max * *width as f64) as usize;
                let empty = *width as usize - filled;
                format!("[{}{}]", "=".repeat(filled), " ".repeat(empty))
            }
            Widget::List { items, selected, .. } => {
                items.iter().enumerate().map(|(i, item)| {
                    let prefix = if i == *selected { "> " } else { "  " };
                    format!("{}{}", prefix, item)
                }).collect::<Vec<_>>().join("\n")
            }
            Widget::Table { headers, rows, .. } => {
                let mut output = headers.join(" | ");
                output.push('\n');
                output.push_str(&"-".repeat(output.len()));
                for row in rows {
                    output.push('\n');
                    output.push_str(&row.join(" | "));
                }
                output
            }
            Widget::Panel { title, children, style } => {
                let inner: String = children.iter().map(|c| c.render()).collect::<Vec<_>>().join("\n");
                if style.border {
                    let width = inner.lines().map(|l| l.len()).max().unwrap_or(20) + 4;
                    let border = "-".repeat(width);
                    format!("+{}+\n| {} |\n|{}|\n+{}+",
                        border, title, inner.lines().map(|l| format!(" {} |", l)).collect::<Vec<_>>().join("\n"), border)
                } else {
                    format!("{}\n{}", title, inner)
                }
            }
            Widget::Container { children } => {
                children.iter().map(|c| c.render()).collect::<Vec<_>>().join("\n")
            }
        }
    }

    pub fn style_mut(&mut self) -> Option<&mut Style> {
        match self {
            Widget::Label { style, .. } => Some(style),
            Widget::Button { style, .. } => Some(style),
            Widget::TextInput { style, .. } => Some(style),
            Widget::CheckBox { style, .. } => Some(style),
            Widget::Radio { style, .. } => Some(style),
            Widget::ProgressBar { style, .. } => Some(style),
            Widget::List { style, .. } => Some(style),
            Widget::Table { style, .. } => Some(style),
            Widget::Panel { style, .. } => Some(style),
            Widget::Container { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label() {
        let w = Widget::label("Hello");
        assert_eq!(w.render(), "Hello");
    }

    #[test]
    fn test_button() {
        let w = Widget::button("Click Me");
        assert_eq!(w.render(), "<Click Me>");
    }

    #[test]
    fn test_checkbox() {
        let w = Widget::checkbox("Accept", true);
        assert_eq!(w.render(), "[x] Accept");
    }

    #[test]
    fn test_progress_bar() {
        let w = Widget::progress_bar(50.0, 100.0);
        let rendered = w.render();
        assert!(rendered.contains("="));
    }

    #[test]
    fn test_list() {
        let w = Widget::list(vec!["Item 1".to_string(), "Item 2".to_string()]);
        let rendered = w.render();
        assert!(rendered.contains("> Item 1"));
    }

    #[test]
    fn test_table() {
        let w = Widget::table(
            vec!["Name".to_string(), "Age".to_string()],
            vec![vec!["Alice".to_string(), "30".to_string()]],
        );
        let rendered = w.render();
        assert!(rendered.contains("Name | Age"));
    }

    #[test]
    fn test_rect() {
        let r = Rect::new(10, 20, 30, 40);
        assert!(r.contains(15, 25));
        assert!(!r.contains(5, 25));
    }
}
