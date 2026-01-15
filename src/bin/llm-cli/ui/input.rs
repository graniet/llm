use ratatui::layout::{Position, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::config::NavigationMode;
use crate::input::InputBuffer;

use super::consts::{INPUT_PADDING, MIN_INPUT_HEIGHT};
use super::theme::{indicators, Theme};

const INPUT_BORDER_OFFSET: u16 = 1;
const PROMPT_WIDTH: u16 = 2; // "❯ " = prompt + space

/// Vi mode indicators
mod mode_labels {
    pub const INSERT: &str = "INS";
    pub const NORMAL: &str = "NOR";
}

pub fn input_height(buffer: &InputBuffer, width: u16) -> u16 {
    let effective_width = width.saturating_sub(INPUT_PADDING + PROMPT_WIDTH);
    let lines = buffer.wrapped_lines(effective_width);
    let height = lines.len() as u16 + INPUT_PADDING;
    height.max(MIN_INPUT_HEIGHT)
}

pub struct InputProps<'a> {
    pub area: Rect,
    pub buffer: &'a InputBuffer,
    pub theme: &'a Theme,
    pub placeholder: &'a str,
    pub show_cursor: bool,
    pub focused: bool,
    pub navigation_mode: NavigationMode,
    pub vi_mode: ViMode,
}

/// Vi editor mode state
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default)]
pub enum ViMode {
    #[default]
    Insert,
    Normal,
}

pub fn render_input(frame: &mut Frame<'_>, props: InputProps<'_>) {
    let inner_width = props
        .area
        .width
        .saturating_sub(INPUT_PADDING + PROMPT_WIDTH);
    let mut lines: Vec<Line<'static>> = Vec::new();

    // Build input lines with prompt
    let wrapped = props.buffer.wrapped_lines(inner_width);

    if wrapped.is_empty() {
        // Placeholder with prompt
        lines.push(build_prompt_line(
            props.placeholder,
            props.theme,
            true,
            props.focused,
        ));
    } else {
        // First line with prompt
        if let Some(first) = wrapped.first() {
            lines.push(build_prompt_line(first, props.theme, false, props.focused));
        }
        // Subsequent lines with continuation indent
        for line in wrapped.iter().skip(1) {
            lines.push(build_continuation_line(line, props.theme));
        }
    }

    // Build the block with border
    let block = build_input_block(
        props.theme,
        props.focused,
        props.navigation_mode,
        props.vi_mode,
    );

    let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    frame.render_widget(paragraph.block(block), props.area);

    // Set cursor position
    if props.show_cursor && !props.buffer.is_empty() {
        let (row, col) = props.buffer.cursor_position(inner_width);
        frame.set_cursor_position(Position::new(
            props.area.x + INPUT_BORDER_OFFSET + PROMPT_WIDTH + col,
            props.area.y + INPUT_BORDER_OFFSET + row,
        ));
    } else if props.show_cursor && props.buffer.is_empty() {
        // Cursor at start after prompt
        frame.set_cursor_position(Position::new(
            props.area.x + INPUT_BORDER_OFFSET + PROMPT_WIDTH,
            props.area.y + INPUT_BORDER_OFFSET,
        ));
    }
}

/// Builds a line with the prompt character
fn build_prompt_line(
    content: &str,
    theme: &Theme,
    is_placeholder: bool,
    focused: bool,
) -> Line<'static> {
    let prompt_style = if focused { theme.prompt } else { theme.muted };

    let content_style = if is_placeholder {
        theme.muted
    } else {
        Style::default()
    };

    Line::from(vec![
        Span::styled(format!("{} ", indicators::PROMPT), prompt_style),
        Span::styled(content.to_string(), content_style),
    ])
}

/// Builds a continuation line (for multi-line input)
fn build_continuation_line(content: &str, theme: &Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled("  ".to_string(), theme.muted), // Indent to align with prompt
        Span::styled(content.to_string(), Style::default()),
    ])
}

/// Builds the input block with border and optional mode indicator
fn build_input_block(
    theme: &Theme,
    focused: bool,
    nav_mode: NavigationMode,
    vi_mode: ViMode,
) -> Block<'static> {
    let border_style = if focused {
        theme.border_focused
    } else {
        theme.border
    };

    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style);

    // Add mode indicator for Vi mode
    if matches!(nav_mode, NavigationMode::Vi) {
        let (mode_label, mode_style) = match vi_mode {
            ViMode::Insert => (mode_labels::INSERT, theme.status_ok),
            ViMode::Normal => (mode_labels::NORMAL, theme.mode_indicator),
        };
        block = block.title_bottom(Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(mode_label.to_string(), mode_style),
            Span::styled(" ", Style::default()),
        ]));
    }

    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_height_minimum() {
        let buffer = InputBuffer::default();
        let height = input_height(&buffer, 80);
        assert!(height >= MIN_INPUT_HEIGHT);
    }

    #[test]
    fn vi_mode_default_is_insert() {
        assert_eq!(ViMode::default(), ViMode::Insert);
    }
}
