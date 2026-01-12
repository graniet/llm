use ratatui::layout::Rect;
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::ToolApprovalState;

use super::super::theme::Theme;

pub fn render_tool_approval(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &ToolApprovalState,
    theme: &Theme,
) {
    let lines = vec![
        Line::from(format!("Tool: {}", state.invocation.name)),
        Line::from(format!("Args: {}", state.invocation.arguments)),
        Line::from("Approve execution? y/n"),
    ]
    .into_iter()
    .map(|line| line.style(theme.status))
    .collect::<Vec<_>>();
    let paragraph = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Tool Approval"),
        )
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}
