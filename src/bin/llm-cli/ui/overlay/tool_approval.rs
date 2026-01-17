use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::ToolApprovalState;

use super::super::theme::Theme;

pub fn render_tool_approval(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &ToolApprovalState,
    theme: &Theme,
) {
    // Calculate a centered area for the dialog
    let dialog_width = 60.min(area.width.saturating_sub(4));
    let dialog_height = 8.min(area.height.saturating_sub(2));
    let dialog_area = centered_rect(dialog_width, dialog_height, area);

    // Clear the area first so the dialog is visible
    frame.render_widget(Clear, dialog_area);

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Tool: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(state.invocation.name.clone(), theme.tool),
        ]),
        Line::from(vec![
            Span::styled("Args: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(truncate_args(
                &state.invocation.arguments,
                dialog_width as usize - 10,
            )),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Approve execution? "),
            Span::styled("[y]", theme.accent),
            Span::raw(" yes  "),
            Span::styled("[n]", theme.accent),
            Span::raw(" no"),
        ]),
    ];

    let paragraph = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_focused)
                .style(Style::default().bg(Color::Black))
                .title(" Tool Approval "),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, dialog_area);
}

fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    Rect::new(x, y, width, height)
}

fn truncate_args(args: &str, max_len: usize) -> String {
    if args.len() <= max_len {
        args.to_string()
    } else {
        format!("{}...", &args[..max_len.saturating_sub(3)])
    }
}
