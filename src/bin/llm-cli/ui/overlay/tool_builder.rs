use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::{ToolBuilderState, ToolBuilderStep};

use super::super::theme::{indicators, Theme};

pub fn render_tool_builder(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &ToolBuilderState,
    theme: &Theme,
) {
    // Clear the area first
    frame.render_widget(Clear, area);
    let mut lines = Vec::new();

    // Title with step indicator
    let step_num = step_number(state.step);
    let total_steps = 5; // Name, Desc, Params, Command, Confirm
    lines.push(Line::from(vec![Span::styled(
        format!("Create Custom Tool ({}/{})", step_num, total_steps),
        theme.accent,
    )]));
    lines.push(Line::default());

    // Show summary of entered data so far
    for summary_line in state.summary_lines() {
        lines.push(Line::from(vec![Span::styled(
            format!("  {}", summary_line),
            theme.muted,
        )]));
    }

    if !state.summary_lines().is_empty() {
        lines.push(Line::default());
    }

    // Current step prompt
    lines.push(Line::from(vec![
        Span::styled(format!("{} ", indicators::PROMPT), theme.prompt),
        Span::styled(state.step_prompt().to_string(), theme.status),
    ]));

    // Input field
    let input_line = Line::from(vec![
        Span::styled("  > ", theme.accent),
        Span::styled(
            state.current_input.clone(),
            theme.status.add_modifier(Modifier::BOLD),
        ),
        Span::styled("_", theme.muted), // Cursor indicator
    ]);
    lines.push(input_line);

    // Error message if any
    if let Some(error) = &state.error {
        lines.push(Line::default());
        lines.push(Line::from(vec![
            Span::styled(format!("  {} ", indicators::CROSS), theme.status_error),
            Span::styled(error.clone(), theme.error),
        ]));
    }

    // Help text
    lines.push(Line::default());
    lines.push(Line::from(vec![Span::styled(
        "  Enter = next step, Esc = cancel",
        theme.muted,
    )]));

    // Show type hints for certain steps
    match state.step {
        ToolBuilderStep::ParamType => {
            lines.push(Line::from(vec![Span::styled(
                "  Types: string, number, boolean (default: string)",
                theme.muted,
            )]));
        }
        ToolBuilderStep::Command => {
            lines.push(Line::from(vec![Span::styled(
                "  Use {{param_name}} to reference parameters",
                theme.muted,
            )]));
        }
        _ => {}
    }

    let text = Text::from(lines);
    let title = format!(" {} Tool Builder ", indicators::BULLET);
    let paragraph = Paragraph::new(text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(theme.border_focused)
                .style(Style::default().bg(Color::Black)),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, area);
}

fn step_number(step: ToolBuilderStep) -> u8 {
    match step {
        ToolBuilderStep::Name => 1,
        ToolBuilderStep::Description => 2,
        ToolBuilderStep::ParamName
        | ToolBuilderStep::ParamType
        | ToolBuilderStep::ParamDesc
        | ToolBuilderStep::ParamRequired
        | ToolBuilderStep::AddMoreParams => 3,
        ToolBuilderStep::Command => 4,
        ToolBuilderStep::Confirm => 5,
    }
}
