/// Advanced GUI framework with event handling, canvas drawing, and theme-aware widgets.

use super::layout::{Layout, LayoutItem};
use super::theme::Theme;
use super::widget::{Color, Rect, Style, Widget};

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// A user-facing event that can be dispatched to the widget tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Mouse click at absolute (x, y).
    Click { x: u16, y: u16 },
    /// A key was pressed.
    KeyPress { key: char },
    /// A text-input value changed.
    ValueChanged { widget_id: String, value: String },
    /// A slider was dragged to a new value.
    SliderChanged { widget_id: String, value: f64 },
    /// Focus moved to a specific widget.
    Focus { widget_id: String },
    /// No-op sentinel.
    None,
}

/// Return value from an event handler.
#[derive(Debug, Clone, PartialEq)]
pub enum EventResult {
    /// Nothing special happened.
    Handled,
    /// A named action was triggered (e.g. a button press).
    Action(String),
    /// Request to close / quit.
    Quit,
}

/// A simple event handler: receives an event and returns a result.
pub trait EventHandler {
    fn handle(&mut self, event: &Event) -> EventResult;
}

/// Event bus that collects handlers and dispatches events to all of them.
pub struct EventBus {
    handlers: Vec<Box<dyn EventHandler>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    pub fn register(&mut self, handler: Box<dyn EventHandler>) {
        self.handlers.push(handler);
    }

    /// Dispatch an event to every registered handler, collecting results.
    pub fn dispatch(&mut self, event: &Event) -> Vec<EventResult> {
        self.handlers
            .iter_mut()
            .map(|h| h.handle(event))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Slider widget (extends the widget system)
// ---------------------------------------------------------------------------

/// A draggable slider that selects a value within a range.
#[derive(Debug, Clone)]
pub struct Slider {
    pub id: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub style: Style,
    pub track_width: u16,
}

impl Slider {
    pub fn new(id: &str, min: f64, max: f64, step: f64) -> Self {
        Self {
            id: id.to_string(),
            value: min,
            min,
            max,
            step,
            style: Style::default(),
            track_width: 20,
        }
    }

    pub fn with_value(mut self, value: f64) -> Self {
        self.value = value.clamp(self.min, self.max);
        self
    }

    pub fn with_style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn set_value(&mut self, value: f64) {
        self.value = value.clamp(self.min, self.max);
    }

    /// Advance the value by `delta` steps (negative to go backwards).
    pub fn nudge(&mut self, delta: i32) {
        self.set_value(self.value + delta as f64 * self.step);
    }

    /// Render the slider as a text string: `[====     ] 42.0`
    pub fn render(&self) -> String {
        let ratio = (self.value - self.min) / (self.max - self.min);
        let filled = (ratio * self.track_width as f64).round() as usize;
        let empty = self.track_width as usize - filled;
        format!(
            "[{}{}] {:.1}",
            "=".repeat(filled),
            " ".repeat(empty),
            self.value
        )
    }
}

// ---------------------------------------------------------------------------
// Canvas drawing primitives
// ---------------------------------------------------------------------------

/// A character-cell canvas that widgets and user code can draw on.
pub struct Canvas {
    width: u16,
    height: u16,
    cells: Vec<Vec<char>>,
    fg_colors: Vec<Vec<Color>>,
    bg_colors: Vec<Vec<Color>>,
}

impl Canvas {
    pub fn new(width: u16, height: u16) -> Self {
        let w = width as usize;
        let h = height as usize;
        Self {
            width,
            height,
            cells: vec![vec![' '; w]; h],
            fg_colors: vec![vec![Color::Default]; w; h],
            bg_colors: vec![vec![Color::Default]; w; h],
        }
    }

    pub fn width(&self) -> u16 {
        self.width
    }
    pub fn height(&self) -> u16 {
        self.height
    }

    /// Clear the entire canvas to spaces.
    pub fn clear(&mut self) {
        for row in self.cells.iter_mut() {
            row.fill(' ');
        }
    }

    /// Put a single character at (x, y).
    pub fn put(&mut self, x: u16, y: u16, ch: char) {
        if x < self.width && y < self.height {
            self.cells[y as usize][x as usize] = ch;
        }
    }

    /// Draw a horizontal line from (x0, y) to (x1, y).
    pub fn hline(&mut self, x0: u16, x1: u16, y: u16, ch: char) {
        for x in x0..=x1.min(self.width.saturating_sub(1)) {
            self.put(x, y, ch);
        }
    }

    /// Draw a vertical line from (x, y0) to (x, y1).
    pub fn vline(&mut self, x: u16, y0: u16, y1: u16, ch: char) {
        for y in y0..=y1.min(self.height.saturating_sub(1)) {
            self.put(x, y, ch);
        }
    }

    /// Draw a rectangle outline.
    pub fn rect(&mut self, r: &Rect) {
        let x1 = r.x + r.width.saturating_sub(1);
        let y1 = r.y + r.height.saturating_sub(1);
        self.hline(r.x, x1, r.y, '-');
        self.hline(r.x, x1, y1, '-');
        self.vline(r.x, r.y, y1, '|');
        self.vline(x1, r.y, y1, '|');
        // corners
        self.put(r.x, r.y, '+');
        self.put(x1, r.y, '+');
        self.put(r.x, y1, '+');
        self.put(x1, y1, '+');
    }

    /// Write a string starting at (x, y), clipping to bounds.
    pub fn text(&mut self, x: u16, y: u16, s: &str) {
        for (i, ch) in s.chars().enumerate() {
            self.put(x + i as u16, y, ch);
        }
    }

    /// Fill a rectangle area with a character.
    pub fn fill_rect(&mut self, r: &Rect, ch: char) {
        for dy in 0..r.height {
            for dx in 0..r.width {
                self.put(r.x + dx, r.y + dy, ch);
            }
        }
    }

    /// Render the canvas into a single `String` (lines separated by `\n`).
    pub fn to_string(&self) -> String {
        self.cells
            .iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ---------------------------------------------------------------------------
// Advanced layout manager (composable wrappers)
// ---------------------------------------------------------------------------

/// Build a horizontal layout from a list of widgets with default flex=1.
pub fn hstack(spacing: u16, widgets: Vec<Widget>) -> Layout {
    let mut layout = Layout::horizontal(spacing);
    for w in widgets {
        layout.add(LayoutItem::new(w));
    }
    layout
}

/// Build a vertical layout from a list of widgets with default flex=1.
pub fn vstack(spacing: u16, widgets: Vec<Widget>) -> Layout {
    let mut layout = Layout::vertical(spacing);
    for w in widgets {
        layout.add(LayoutItem::new(w));
    }
    layout
}

/// Build a grid layout with the given number of columns.
pub fn grid(columns: usize, spacing: u16, widgets: Vec<Widget>) -> Layout {
    let mut layout = Layout::grid(columns, spacing);
    for w in widgets {
        layout.add(LayoutItem::new(w));
    }
    layout
}

// ---------------------------------------------------------------------------
// Theme-aware rendering helpers
// ---------------------------------------------------------------------------

/// Render a widget using the appropriate style from the theme.
pub fn themed_widget(widget: &Widget, theme: &Theme) -> String {
    let _ = theme; // theme styles are terminal-ANSI in a real TUI; here we
                   // just delegate to the widget's own render for the text
                   // representation.
    widget.render()
}

/// Apply a theme's button style to a mutable widget, if applicable.
pub fn apply_theme(widget: &mut Widget, theme: &Theme) {
    if let Some(style) = widget.style_mut() {
        *style = theme.button_style.clone();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Slider tests --

    #[test]
    fn slider_default_is_min() {
        let s = Slider::new("vol", 0.0, 100.0, 1.0);
        assert_eq!(s.value, 0.0);
    }

    #[test]
    fn slider_clamp_to_range() {
        let s = Slider::new("vol", 0.0, 100.0, 1.0).with_value(150.0);
        assert_eq!(s.value, 100.0);
    }

    #[test]
    fn slider_nudge() {
        let mut s = Slider::new("vol", 0.0, 100.0, 10.0);
        s.nudge(3);
        assert_eq!(s.value, 30.0);
        s.nudge(-2);
        assert_eq!(s.value, 10.0);
    }

    #[test]
    fn slider_render_contains_equals() {
        let s = Slider::new("x", 0.0, 10.0, 1.0).with_value(5.0);
        let out = s.render();
        assert!(out.contains('='));
        assert!(out.contains('5'));
    }

    // -- Event system tests --

    struct TestHandler {
        last: Option<Event>,
    }

    impl TestHandler {
        fn new() -> Self {
            Self { last: None }
        }
    }

    impl EventHandler for TestHandler {
        fn handle(&mut self, event: &Event) -> EventResult {
            self.last = Some(event.clone());
            match event {
                Event::KeyPress { key: 'q' } => EventResult::Quit,
                Event::Click { .. } => EventResult::Action("clicked".into()),
                _ => EventResult::Handled,
            }
        }
    }

    #[test]
    fn event_bus_dispatches_to_all() {
        let mut bus = EventBus::new();
        bus.register(Box::new(TestHandler::new()));
        bus.register(Box::new(TestHandler::new()));

        let results = bus.dispatch(&Event::KeyPress { key: 'q' });
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], EventResult::Quit);
    }

    #[test]
    fn event_bus_returns_action_for_click() {
        let mut bus = EventBus::new();
        bus.register(Box::new(TestHandler::new()));

        let results = bus.dispatch(&Event::Click { x: 5, y: 10 });
        assert_eq!(results[0], EventResult::Action("clicked".into()));
    }

    // -- Canvas tests --

    #[test]
    fn canvas_put_and_to_string() {
        let mut c = Canvas::new(5, 1);
        c.put(0, 0, 'H');
        c.put(1, 0, 'i');
        assert_eq!(c.to_string(), "Hi   ");
    }

    #[test]
    fn canvas_hline() {
        let mut c = Canvas::new(8, 1);
        c.hline(1, 5, 0, '-');
        assert_eq!(c.to_string(), " ----- ");
    }

    #[test]
    fn canvas_rect() {
        let mut c = Canvas::new(6, 3);
        c.rect(&Rect::new(0, 0, 6, 3));
        let out = c.to_string();
        assert!(out.contains('+'));
        assert!(out.contains('-'));
        assert!(out.contains('|'));
    }

    #[test]
    fn canvas_text() {
        let mut c = Canvas::new(10, 1);
        c.text(2, 0, "hi");
        assert_eq!(c.to_string(), "  hi      ");
    }

    #[test]
    fn canvas_fill_rect() {
        let mut c = Canvas::new(4, 2);
        c.fill_rect(&Rect::new(1, 0, 2, 2), '#');
        let out = c.to_string();
        assert!(out.contains('#'));
    }

    #[test]
    fn canvas_clear() {
        let mut c = Canvas::new(3, 1);
        c.text(0, 0, "abc");
        c.clear();
        assert_eq!(c.to_string(), "   ");
    }

    // -- Layout helper tests --

    #[test]
    fn hstack_produces_horizontal() {
        let layout = hstack(1, vec![Widget::label("A"), Widget::label("B")]);
        let positions = layout.compute(&Rect::new(0, 0, 100, 10));
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn vstack_produces_vertical() {
        let layout = vstack(1, vec![Widget::label("A"), Widget::label("B")]);
        let positions = layout.compute(&Rect::new(0, 0, 100, 100));
        assert_eq!(positions.len(), 2);
        assert!(positions[0].0.y < positions[1].0.y);
    }

    #[test]
    fn grid_helper_works() {
        let layout = grid(2, 0, vec![
            Widget::label("A"),
            Widget::label("B"),
            Widget::label("C"),
            Widget::label("D"),
        ]);
        let positions = layout.compute(&Rect::new(0, 0, 100, 100));
        assert_eq!(positions.len(), 4);
    }

    // -- Theme integration tests --

    #[test]
    fn themed_widget_returns_rendered_text() {
        let theme = Theme::dark();
        let w = Widget::button("OK");
        assert_eq!(themed_widget(&w, &theme), "<OK>");
    }

    #[test]
    fn apply_theme_sets_style() {
        let theme = Theme::monokai();
        let mut w = Widget::button("Go");
        apply_theme(&mut w, &theme);
        let s = w.style_mut().unwrap();
        assert!(s.bold);
    }
}
