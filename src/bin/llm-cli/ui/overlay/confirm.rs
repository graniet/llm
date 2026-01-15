use ratatui::layout::Rect;
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::ConfirmState;

use super::super::theme::Theme;

pub fn render_confirm(frame: &mut Frame<'_>, area: Rect, state: &ConfirmState, theme: &Theme) {
    let lines = vec![
        Line::from(state.message.clone()),
        Line::from("y = confirm, n = cancel"),
    ]
    .into_iter()
    .map(|line| line.style(theme.status))
    .collect::<Vec<_>>();
    let paragraph = Paragraph::new(Text::from(lines))
        .block(Block::default().borders(Borders::ALL).title("Confirm"))
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}
