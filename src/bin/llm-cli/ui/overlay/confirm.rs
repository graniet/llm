use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::ConfirmState;

use super::super::theme::Theme;

pub fn render_confirm(frame: &mut Frame<'_>, area: Rect, state: &ConfirmState, theme: &Theme) {
    // Calculate a smaller centered area for the confirm dialog
    let dialog_width = 40.min(area.width.saturating_sub(4));
    let dialog_height = 5.min(area.height.saturating_sub(2));
    let dialog_area = centered_rect(dialog_width, dialog_height, area);

    // Clear the area first so the dialog is visible
    frame.render_widget(Clear, dialog_area);

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            state.message.clone(),
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("[y]", theme.accent),
            Span::raw(" confirm  "),
            Span::styled("[n]", theme.accent),
            Span::raw(" cancel"),
        ]),
    ];

    let paragraph = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_focused)
                .style(Style::default().bg(Color::Black))
                .title(" Confirm "),
        )
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, dialog_area);
}

fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    Rect::new(x, y, width, height)
}
