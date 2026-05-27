/// Theme system for GUI styling.

use super::widget::{Style, Color};

#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    pub primary: Color,
    pub secondary: Color,
    pub accent: Color,
    pub background: Color,
    pub foreground: Color,
    pub success: Color,
    pub warning: Color,
    pub error: Color,
    pub border: Color,
    pub title_style: Style,
    pub body_style: Style,
    pub button_style: Style,
    pub input_style: Style,
}

impl Theme {
    pub fn default_theme() -> Self {
        Self {
            name: "Default".to_string(),
            primary: Color::Blue,
            secondary: Color::Cyan,
            accent: Color::Magenta,
            background: Color::Default,
            foreground: Color::Default,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            border: Color::Default,
            title_style: Style { bold: true, ..Style::default() },
            body_style: Style::default(),
            button_style: Style { bold: true, ..Style::default() },
            input_style: Style::default(),
        }
    }

    pub fn dark() -> Self {
        Self {
            name: "Dark".to_string(),
            primary: Color::Cyan,
            secondary: Color::Blue,
            accent: Color::Magenta,
            background: Color::Black,
            foreground: Color::White,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            border: Color::White,
            title_style: Style { fg: Color::White, bg: Color::Black, bold: true, ..Style::default() },
            body_style: Style { fg: Color::White, bg: Color::Black, ..Style::default() },
            button_style: Style { fg: Color::Black, bg: Color::Cyan, bold: true, ..Style::default() },
            input_style: Style { fg: Color::White, bg: Color::Black, ..Style::default() },
        }
    }

    pub fn light() -> Self {
        Self {
            name: "Light".to_string(),
            primary: Color::Blue,
            secondary: Color::Cyan,
            accent: Color::Magenta,
            background: Color::White,
            foreground: Color::Black,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            border: Color::Black,
            title_style: Style { fg: Color::Black, bg: Color::White, bold: true, ..Style::default() },
            body_style: Style { fg: Color::Black, bg: Color::White, ..Style::default() },
            button_style: Style { fg: Color::White, bg: Color::Blue, bold: true, ..Style::default() },
            input_style: Style { fg: Color::Black, bg: Color::White, ..Style::default() },
        }
    }

    pub fn monokai() -> Self {
        Self {
            name: "Monokai".to_string(),
            primary: Color::Green,
            secondary: Color::Yellow,
            accent: Color::Magenta,
            background: Color::Black,
            foreground: Color::White,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            border: Color::Green,
            title_style: Style { fg: Color::Green, bg: Color::Black, bold: true, ..Style::default() },
            body_style: Style { fg: Color::White, bg: Color::Black, ..Style::default() },
            button_style: Style { fg: Color::Black, bg: Color::Green, bold: true, ..Style::default() },
            input_style: Style { fg: Color::White, bg: Color::Black, ..Style::default() },
        }
    }

    pub fn solarized() -> Self {
        Self {
            name: "Solarized".to_string(),
            primary: Color::Blue,
            secondary: Color::Cyan,
            accent: Color::Yellow,
            background: Color::Black,
            foreground: Color::Cyan,
            success: Color::Green,
            warning: Color::Yellow,
            error: Color::Red,
            border: Color::Blue,
            title_style: Style { fg: Color::Cyan, bg: Color::Black, bold: true, ..Style::default() },
            body_style: Style { fg: Color::Cyan, bg: Color::Black, ..Style::default() },
            button_style: Style { fg: Color::Black, bg: Color::Blue, bold: true, ..Style::default() },
            input_style: Style { fg: Color::Cyan, bg: Color::Black, ..Style::default() },
        }
    }

    pub fn title_style(&self) -> &Style {
        &self.title_style
    }

    pub fn body_style(&self) -> &Style {
        &self.body_style
    }

    pub fn button_style(&self) -> &Style {
        &self.button_style
    }

    pub fn input_style(&self) -> &Style {
        &self.input_style
    }

    pub fn style_for_status(&self, status: &str) -> Style {
        match status {
            "success" => Style { fg: self.success, ..Style::default() },
            "warning" => Style { fg: self.warning, ..Style::default() },
            "error" => Style { fg: self.error, ..Style::default() },
            _ => Style::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_theme() {
        let theme = Theme::default_theme();
        assert_eq!(theme.name, "Default");
    }

    #[test]
    fn test_dark_theme() {
        let theme = Theme::dark();
        assert_eq!(theme.background, Color::Black);
        assert_eq!(theme.foreground, Color::White);
    }

    #[test]
    fn test_theme_styles() {
        let theme = Theme::monokai();
        assert!(theme.title_style().bold);
    }

    #[test]
    fn test_status_styles() {
        let theme = Theme::default_theme();
        let success_style = theme.style_for_status("success");
        assert_eq!(success_style.fg, Color::Green);
    }
}
