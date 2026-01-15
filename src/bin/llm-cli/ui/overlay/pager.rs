use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::PagerState;

use super::super::theme::Theme;

const PAGER_HELP_HEIGHT: u16 = 1;

pub fn render_pager(frame: &mut Frame<'_>, area: Rect, state: &PagerState, theme: &Theme) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(state.title.as_str());
    frame.render_widget(block.clone(), area);
    let inner = block.inner(area);
    if inner.height == 0 {
        return;
    }
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(PAGER_HELP_HEIGHT)])
        .split(inner);
    let text = Text::from(
        state
            .lines
            .iter()
            .map(|line| Line::from(line.clone()))
            .collect::<Vec<_>>(),
    );
    let content = Paragraph::new(text)
        .style(theme.assistant)
        .wrap(Wrap { trim: false })
        .scroll((state.scroll, 0));
    frame.render_widget(content, chunks[0]);
    let help = Paragraph::new("Esc to close · j/k to scroll · g/G top/bottom").style(theme.muted);
    frame.render_widget(help, chunks[1]);
}
