use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::SlashCommandState;

use super::super::theme::Theme;

const QUERY_HEIGHT: u16 = 1;

pub fn render_slash_popup(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &SlashCommandState,
    theme: &Theme,
) {
    // Clear the area first
    frame.render_widget(Clear, area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme.border_focused)
        .style(Style::default().bg(Color::Black))
        .title(" Commands ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    render_query(frame, inner, state, theme);
    render_list(frame, inner, state, theme);
}

fn render_query(frame: &mut Frame<'_>, inner: Rect, state: &SlashCommandState, theme: &Theme) {
    let query = format!("/{}", state.query);
    let query_line = Line::from(vec![
        Span::styled("Filter: ", theme.muted),
        Span::raw(query),
    ]);
    let query_area = Rect::new(inner.x, inner.y, inner.width, QUERY_HEIGHT);
    frame.render_widget(
        Paragraph::new(query_line).wrap(Wrap { trim: true }),
        query_area,
    );
}

fn render_list(frame: &mut Frame<'_>, inner: Rect, state: &SlashCommandState, theme: &Theme) {
    let list_area = Rect::new(
        inner.x,
        inner.y + QUERY_HEIGHT,
        inner.width,
        inner.height.saturating_sub(QUERY_HEIGHT),
    );
    let items: Vec<ListItem> = state
        .filtered
        .iter()
        .enumerate()
        .map(|(idx, cmd)| build_item(idx, cmd, state, theme))
        .collect();
    frame.render_widget(List::new(items), list_area);
}

fn build_item(
    idx: usize,
    cmd: &crate::runtime::SlashCommand,
    state: &SlashCommandState,
    theme: &Theme,
) -> ListItem<'static> {
    let mut spans = vec![
        Span::styled(format!("/{}", cmd.name), theme.accent),
        Span::raw("  "),
        Span::raw(cmd.description),
        Span::styled(format!("  [{}]", cmd.category.label()), theme.muted),
    ];
    if let Some(shortcut) = cmd.shortcut {
        spans.push(Span::styled(format!("  {shortcut}"), theme.muted));
    }
    let mut style = Style::default();
    if idx == state.selected {
        style = style.add_modifier(Modifier::REVERSED);
    }
    ListItem::new(Line::from(spans)).style(style)
}
