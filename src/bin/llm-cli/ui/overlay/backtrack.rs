use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::BacktrackOverlayState;

use super::super::theme::Theme;

const INDEX_WIDTH: usize = 3;
const HEADER_HEIGHT: u16 = 2;

pub fn render_backtrack(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &BacktrackOverlayState,
    theme: &Theme,
) {
    let block = Block::default().borders(Borders::ALL).title("Backtrack");
    frame.render_widget(block.clone(), area);
    let inner = block.inner(area);
    if inner.height == 0 {
        return;
    }
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(HEADER_HEIGHT), Constraint::Min(1)])
        .split(inner);
    let header = selected_header(state).unwrap_or_else(|| "No snapshots yet".to_string());
    let header_paragraph = Paragraph::new(header).style(theme.muted);
    frame.render_widget(header_paragraph, chunks[0]);
    let lines = build_snapshot_lines(state, theme);
    let list = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    frame.render_widget(list, chunks[1]);
}

fn selected_header(state: &BacktrackOverlayState) -> Option<String> {
    let selected = state.entries.get(state.selected)?;
    Some(format!(
        "{} · {} msgs · {}",
        selected.title,
        selected.message_count,
        selected.created_at.format("%H:%M:%S")
    ))
}

fn build_snapshot_lines(state: &BacktrackOverlayState, theme: &Theme) -> Vec<Line<'static>> {
    state
        .entries
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            let label = format!(
                "{:>width$} {} · {} msgs",
                idx + 1,
                entry.created_at.format("%H:%M:%S"),
                entry.message_count,
                width = INDEX_WIDTH
            );
            if idx == state.selected {
                Line::styled(label, Style::default().add_modifier(Modifier::REVERSED))
            } else {
                Line::styled(label, theme.assistant)
            }
        })
        .collect()
}
