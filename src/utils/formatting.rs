pub fn format_number(n: f64, decimals: usize) -> String {
    format!("{:.1$}", n, decimals)
}

pub fn format_with_separators(n: i64) -> String {
    let s = n.to_string();
    let negative = s.starts_with('-');
    let digits = if negative { &s[1..] } else { &s };
    let mut result = String::new();
    for (i, c) in digits.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    if negative {
        result.push('-');
    }
    result.chars().rev().collect()
}

pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

pub fn format_duration_ms(ms: f64) -> String {
    if ms < 1.0 {
        format!("{:.2} µs", ms * 1000.0)
    } else if ms < 1000.0 {
        format!("{:.2} ms", ms)
    } else if ms < 60000.0 {
        format!("{:.2} s", ms / 1000.0)
    } else if ms < 3600000.0 {
        format!("{:.2} min", ms / 60000.0)
    } else {
        format!("{:.2} hr", ms / 3600000.0)
    }
}

pub fn format_percentage(value: f64, total: f64) -> String {
    if total == 0.0 {
        return "0%".to_string();
    }
    format!("{:.1}%", value / total * 100.0)
}

pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

pub fn indent(s: &str, spaces: usize) -> String {
    let prefix = " ".repeat(spaces);
    s.lines()
        .map(|line| format!("{}{}", prefix, line))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn dedent(s: &str) -> String {
    let min_indent = s.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    s.lines()
        .map(|line| {
            if line.len() >= min_indent {
                &line[min_indent..]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn word_wrap(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in s.split_whitespace() {
        if current_line.is_empty() {
            current_line.push_str(word);
        } else if current_line.len() + 1 + word.len() <= width {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(current_line);
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines
}

pub fn table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.len());
            }
        }
    }

    let mut output = String::new();

    // Header
    for (i, header) in headers.iter().enumerate() {
        if i > 0 {
            output.push_str(" | ");
        }
        output.push_str(&format!("{:<width$}", header, width = widths[i]));
    }
    output.push('\n');

    // Separator
    for (i, width) in widths.iter().enumerate() {
        if i > 0 {
            output.push_str("-+-");
        }
        output.push_str(&"-".repeat(*width));
    }
    output.push('\n');

    // Rows
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i > 0 {
                output.push_str(" | ");
            }
            let width = widths.get(i).copied().unwrap_or(0);
            output.push_str(&format!("{:<width$}", cell, width = width));
        }
        output.push('\n');
    }

    output
}
