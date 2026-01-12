use ratatui::layout::Rect;
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use super::super::theme::Theme;

pub fn render_help(frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
    let lines = vec![
        Line::from("Navigation: j/k or arrows, g/G, Ctrl+u/d"),
        Line::from("Input (simple): Enter send, Shift+Enter newline"),
        Line::from("Input (vi): i insert, Esc normal, Enter newline"),
        Line::from("Focus: Tab toggle input/messages"),
        Line::from("Actions: Ctrl+n new, Ctrl+f fork, Ctrl+s save, Ctrl+p provider"),
        Line::from("Messages: Enter expand/pager, p pager, D diff, y copy, d delete"),
        Line::from("History: Ctrl+z backtrack"),
        Line::from("Commands: / open command palette"),
        Line::from("Search: / (vi mode), Esc to close"),
        Line::from("Exit: Ctrl+c"),
    ]
    .into_iter()
    .map(|line| line.style(theme.status))
    .collect::<Vec<_>>();
    let paragraph = Paragraph::new(Text::from(lines))
        .block(Block::default().borders(Borders::ALL).title("Help"))
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}
