use ratatui::style::{Color, Modifier, Style, Stylize};

#[derive(Debug, Clone)]
pub struct MarkdownStyles {
    pub h1: Style,
    pub h2: Style,
    pub h3: Style,
    pub code: Style,
    pub emphasis: Style,
    pub strong: Style,
    pub strikethrough: Style,
    pub list_marker: Style,
    pub link: Style,
    pub blockquote: Style,
    // Code block specific styles
    pub code_bg: Style,
    pub code_border: Style,
    pub code_header: Style,
}

impl Default for MarkdownStyles {
    fn default() -> Self {
        // Warm color palette matching the theme
        let orange = Color::Rgb(217, 119, 87);
        let dim_gray = Color::Rgb(100, 95, 90);
        let muted_gray = Color::Rgb(140, 135, 130);
        let code_bg_color = Color::Rgb(35, 35, 40);

        Self {
            h1: Style::new().bold().underlined(),
            h2: Style::new().bold(),
            h3: Style::new().bold().italic(),
            code: Style::new()
                .fg(orange)
                .bg(code_bg_color)
                .add_modifier(Modifier::BOLD),
            emphasis: Style::new().italic(),
            strong: Style::new().bold(),
            strikethrough: Style::new().crossed_out(),
            list_marker: Style::new().fg(orange),
            link: Style::new().fg(orange).underlined(),
            blockquote: Style::new().fg(muted_gray).italic(),
            // Code block styles
            code_bg: Style::new().bg(code_bg_color),
            code_border: Style::new().fg(dim_gray),
            code_header: Style::new().fg(muted_gray).add_modifier(Modifier::DIM),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_styles_are_valid() {
        let styles = MarkdownStyles::default();
        // Just verify they can be created without panic
        assert_ne!(styles.code, Style::default());
        assert_ne!(styles.code_bg, Style::default());
    }
}
