use ratatui::layout::Rect;
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::SearchState;

use super::super::theme::Theme;

pub fn render_search(frame: &mut Frame<'_>, area: Rect, state: &SearchState, theme: &Theme) {
    let summary = format!("Matches: {}  (Enter to jump)", state.matches.len());
    let lines = vec![Line::from(format!("/{}", state.query)), Line::from(summary)]
        .into_iter()
        .map(|line| line.style(theme.status))
        .collect::<Vec<_>>();
    let paragraph = Paragraph::new(Text::from(lines))
        .block(Block::default().borders(Borders::ALL).title("Search"))
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}
