/// Layout system for GUI widgets.

use super::widget::{Widget, Rect};

#[derive(Debug, Clone)]
pub enum Layout {
    Horizontal {
        spacing: u16,
        children: Vec<LayoutItem>,
    },
    Vertical {
        spacing: u16,
        children: Vec<LayoutItem>,
    },
    Grid {
        columns: usize,
        spacing: u16,
        children: Vec<LayoutItem>,
    },
    Stack {
        children: Vec<LayoutItem>,
    },
    Center {
        child: Box<LayoutItem>,
    },
    Padding {
        top: u16,
        right: u16,
        bottom: u16,
        left: u16,
        child: Box<LayoutItem>,
    },
}

#[derive(Debug, Clone)]
pub struct LayoutItem {
    pub widget: Widget,
    pub flex: f64,
    pub min_width: Option<u16>,
    pub min_height: Option<u16>,
    pub max_width: Option<u16>,
    pub max_height: Option<u16>,
}

impl LayoutItem {
    pub fn new(widget: Widget) -> Self {
        Self {
            widget,
            flex: 1.0,
            min_width: None,
            min_height: None,
            max_width: None,
            max_height: None,
        }
    }

    pub fn flex(mut self, flex: f64) -> Self {
        self.flex = flex;
        self
    }

    pub fn min_width(mut self, width: u16) -> Self {
        self.min_width = Some(width);
        self
    }

    pub fn min_height(mut self, height: u16) -> Self {
        self.min_height = Some(height);
        self
    }
}

impl Layout {
    pub fn horizontal(spacing: u16) -> Self {
        Layout::Horizontal {
            spacing,
            children: Vec::new(),
        }
    }

    pub fn vertical(spacing: u16) -> Self {
        Layout::Vertical {
            spacing,
            children: Vec::new(),
        }
    }

    pub fn grid(columns: usize, spacing: u16) -> Self {
        Layout::Grid {
            columns,
            spacing,
            children: Vec::new(),
        }
    }

    pub fn add(&mut self, item: LayoutItem) {
        match self {
            Layout::Horizontal { children, .. } => children.push(item),
            Layout::Vertical { children, .. } => children.push(item),
            Layout::Grid { children, .. } => children.push(item),
            Layout::Stack { children } => children.push(item),
            _ => {}
        }
    }

    pub fn compute(&self, bounds: &Rect) -> Vec<(Rect, &Widget)> {
        match self {
            Layout::Horizontal { spacing, children } => {
                self.compute_horizontal(bounds, *spacing, children)
            }
            Layout::Vertical { spacing, children } => {
                self.compute_vertical(bounds, *spacing, children)
            }
            Layout::Grid { columns, spacing, children } => {
                self.compute_grid(bounds, *columns, *spacing, children)
            }
            Layout::Stack { children } => {
                children.iter().map(|item| (*bounds, &item.widget)).collect()
            }
            Layout::Center { child } => {
                let inner = Rect::new(
                    bounds.x + bounds.width / 4,
                    bounds.y + bounds.height / 4,
                    bounds.width / 2,
                    bounds.height / 2,
                );
                vec![(inner, &child.widget)]
            }
            Layout::Padding { top, right, bottom, left, child } => {
                let inner = Rect::new(
                    bounds.x + left,
                    bounds.y + top,
                    bounds.width - left - right,
                    bounds.height - top - bottom,
                );
                vec![(inner, &child.widget)]
            }
        }
    }

    fn compute_horizontal(&self, bounds: &Rect, spacing: u16, children: &[LayoutItem]) -> Vec<(Rect, &Widget)> {
        let total_flex: f64 = children.iter().map(|c| c.flex).sum();
        let total_spacing = spacing * children.len().saturating_sub(1) as u16;
        let available = bounds.width.saturating_sub(total_spacing);

        let mut result = Vec::new();
        let mut x = bounds.x;

        for child in children {
            let width = (available as f64 * child.flex / total_flex) as u16;
            let rect = Rect::new(x, bounds.y, width, bounds.height);
            result.push((rect, &child.widget));
            x += width + spacing;
        }

        result
    }

    fn compute_vertical(&self, bounds: &Rect, spacing: u16, children: &[LayoutItem]) -> Vec<(Rect, &Widget)> {
        let total_flex: f64 = children.iter().map(|c| c.flex).sum();
        let total_spacing = spacing * children.len().saturating_sub(1) as u16;
        let available = bounds.height.saturating_sub(total_spacing);

        let mut result = Vec::new();
        let mut y = bounds.y;

        for child in children {
            let height = (available as f64 * child.flex / total_flex) as u16;
            let rect = Rect::new(bounds.x, y, bounds.width, height);
            result.push((rect, &child.widget));
            y += height + spacing;
        }

        result
    }

    fn compute_grid(&self, bounds: &Rect, columns: usize, spacing: u16, children: &[LayoutItem]) -> Vec<(Rect, &Widget)> {
        let cell_width = (bounds.width - spacing * (columns as u16 - 1)) / columns as u16;
        let rows = (children.len() + columns - 1) / columns;
        let cell_height = (bounds.height - spacing * (rows as u16 - 1)) / rows as u16;

        let mut result = Vec::new();
        for (i, child) in children.iter().enumerate() {
            let col = i % columns;
            let row = i / columns;
            let x = bounds.x + (col as u16) * (cell_width + spacing);
            let y = bounds.y + (row as u16) * (cell_height + spacing);
            result.push((Rect::new(x, y, cell_width, cell_height), &child.widget));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizontal_layout() {
        let mut layout = Layout::horizontal(1);
        layout.add(LayoutItem::new(Widget::label("A")));
        layout.add(LayoutItem::new(Widget::label("B")));

        let bounds = Rect::new(0, 0, 100, 10);
        let positions = layout.compute(&bounds);
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn test_vertical_layout() {
        let mut layout = Layout::vertical(1);
        layout.add(LayoutItem::new(Widget::label("A")));
        layout.add(LayoutItem::new(Widget::label("B")));

        let bounds = Rect::new(0, 0, 100, 100);
        let positions = layout.compute(&bounds);
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn test_grid_layout() {
        let mut layout = Layout::grid(2, 1);
        layout.add(LayoutItem::new(Widget::label("A")));
        layout.add(LayoutItem::new(Widget::label("B")));
        layout.add(LayoutItem::new(Widget::label("C")));
        layout.add(LayoutItem::new(Widget::label("D")));

        let bounds = Rect::new(0, 0, 100, 100);
        let positions = layout.compute(&bounds);
        assert_eq!(positions.len(), 4);
    }

    #[test]
    fn test_padding_layout() {
        let layout = Layout::Padding {
            top: 2,
            right: 2,
            bottom: 2,
            left: 2,
            child: Box::new(LayoutItem::new(Widget::label("Hello"))),
        };

        let bounds = Rect::new(0, 0, 100, 100);
        let positions = layout.compute(&bounds);
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].0.x, 2);
        assert_eq!(positions[0].0.y, 2);
    }
}
