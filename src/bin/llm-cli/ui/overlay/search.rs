use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::SearchState;

use super::super::theme::Theme;

pub fn render_search(frame: &mut Frame<'_>, area: Rect, state: &SearchState, theme: &Theme) {
    // Calculate a smaller centered area for the search dialog
    let dialog_width = 50.min(area.width.saturating_sub(4));
    let dialog_height = 4.min(area.height.saturating_sub(2));
    let dialog_area = centered_rect(dialog_width, dialog_height, area);

    // Clear the area first
    frame.render_widget(Clear, dialog_area);

    let lines = vec![
        Line::from(vec![
            Span::styled("/", theme.accent),
            Span::raw(state.query.clone()),
        ]),
        Line::from(vec![
            Span::raw("Matches: "),
            Span::styled(state.matches.len().to_string(), theme.accent),
            Span::raw("  (Enter to jump)"),
        ]),
    ];

    let paragraph = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_focused)
                .style(Style::default().bg(Color::Black))
                .title(" Search "),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, dialog_area);
}

fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    Rect::new(x, y, width, height)
}
