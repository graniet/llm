use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::PickerState;

use super::super::theme::Theme;

const QUERY_HEIGHT: u16 = 1;

pub fn render_picker(frame: &mut Frame<'_>, area: Rect, state: &PickerState, theme: &Theme) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(state.title.clone());
    let inner = block.inner(area);
    frame.render_widget(block, area);
    render_query(frame, inner, state, theme);
    render_list(frame, inner, state, theme);
}

fn render_query(frame: &mut Frame<'_>, inner: Rect, state: &PickerState, theme: &Theme) {
    let query_line = Line::from(vec![
        Span::styled("Search: ", theme.muted),
        Span::raw(state.query.clone()),
    ]);
    let query_area = Rect::new(inner.x, inner.y, inner.width, QUERY_HEIGHT);
    frame.render_widget(
        Paragraph::new(query_line).wrap(Wrap { trim: true }),
        query_area,
    );
}

fn render_list(frame: &mut Frame<'_>, inner: Rect, state: &PickerState, theme: &Theme) {
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
        .map(|(idx, item)| build_item(idx, item, state, theme))
        .collect();
    frame.render_widget(List::new(items), list_area);
}

fn build_item(
    idx: usize,
    item: &crate::runtime::PickerItem,
    state: &PickerState,
    theme: &Theme,
) -> ListItem<'static> {
    let mut spans = vec![Span::raw(item.label.clone())];
    if let Some(meta) = &item.meta {
        spans.push(Span::styled(format!("  {meta}"), theme.muted));
    }
    if !item.badges.is_empty() {
        let badges = item.badges.join(" ");
        spans.push(Span::styled(format!("  [{badges}]"), theme.accent));
    }
    let mut style = Style::default();
    if idx == state.selected {
        style = style.add_modifier(Modifier::REVERSED);
    }
    ListItem::new(Text::from(Line::from(spans))).style(style)
}
