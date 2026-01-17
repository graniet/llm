//! Dialogue builder overlay rendering.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::{DialogueBuilderState, DialogueBuilderStep, ParticipantField};
use crate::ui::theme::Theme;

pub fn render_dialogue_builder(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &DialogueBuilderState,
    theme: &Theme,
) {
    // Center the overlay
    let dialog_width = area.width.min(70);
    let dialog_height = area.height.min(24);
    let x = (area.width.saturating_sub(dialog_width)) / 2;
    let y = (area.height.saturating_sub(dialog_height)) / 2;
    let dialog_area = Rect::new(x, y, dialog_width, dialog_height);

    // Clear the area
    frame.render_widget(Clear, dialog_area);

    // Main block
    let title = format!(" Dialogue Builder - {} ", state.step.display_name());
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(theme.border_focused)
        .style(Style::default().bg(Color::Black));

    let inner = block.inner(dialog_area);
    frame.render_widget(block, dialog_area);

    // Layout: content area and footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(inner);

    // Render step-specific content
    match state.step {
        DialogueBuilderStep::Participants => {
            render_participants_step(frame, chunks[0], state, theme);
        }
        DialogueBuilderStep::ConfigureParticipant => {
            render_configure_participant_step(frame, chunks[0], state, theme);
        }
        DialogueBuilderStep::Mode => {
            render_mode_step(frame, chunks[0], state, theme);
        }
        DialogueBuilderStep::InitialPrompt => {
            render_prompt_step(frame, chunks[0], state, theme);
        }
        DialogueBuilderStep::Review => {
            render_review_step(frame, chunks[0], state, theme);
        }
    }

    // Render footer with error or instructions
    render_footer(frame, chunks[1], state, theme);
}

fn render_participants_step(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &DialogueBuilderState,
    theme: &Theme,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Input
            Constraint::Length(1), // Label
            Constraint::Min(1),    // List
        ])
        .split(area);

    // Input for adding participants
    let input_block = Block::default()
        .title(" Add Participant (provider:model) ")
        .borders(Borders::ALL)
        .border_style(theme.border);

    let input_text = if state.input.is_empty() {
        Paragraph::new("@openai:gpt-4").style(theme.muted)
    } else {
        Paragraph::new(state.input.as_str()).style(theme.assistant)
    };

    frame.render_widget(input_block, chunks[0]);
    let input_area = Rect::new(
        chunks[0].x + 1,
        chunks[0].y + 1,
        chunks[0].width.saturating_sub(2),
        1,
    );
    frame.render_widget(input_text, input_area);

    // Label
    let label = Paragraph::new("Participants:").style(theme.accent);
    frame.render_widget(label, chunks[1]);

    // List of participants
    let items: Vec<ListItem> = state
        .participants
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (r, g, b) = p.color.rgb();
            let color_style = Style::default().fg(Color::Rgb(r, g, b));
            let style = if i == state.selected {
                color_style.add_modifier(Modifier::REVERSED)
            } else {
                color_style
            };
            ListItem::new(format!(
                "  {} ({}:{})",
                p.display_name, p.provider_id, p.model_id
            ))
            .style(style)
        })
        .collect();

    if items.is_empty() {
        let empty = Paragraph::new("  No participants yet. Add at least 2.").style(theme.muted);
        frame.render_widget(empty, chunks[2]);
    } else {
        let list = List::new(items);
        frame.render_widget(list, chunks[2]);
    }
}

fn render_configure_participant_step(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &DialogueBuilderState,
    theme: &Theme,
) {
    let Some(participant) = state.current_editing_participant() else {
        return;
    };

    let (r, g, b) = participant.color.rgb();
    let participant_color = Color::Rgb(r, g, b);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Header
            Constraint::Length(4), // Display name field
            Constraint::Length(6), // System prompt field
            Constraint::Min(1),    // Spacer
        ])
        .split(area);

    // Header showing which participant
    let header = Paragraph::new(Line::from(vec![
        Span::styled("Configuring: ", theme.accent),
        Span::styled(
            format!(
                "{} ({}:{})",
                participant.display_name, participant.provider_id, participant.model_id
            ),
            Style::default().fg(participant_color),
        ),
    ]));
    frame.render_widget(header, chunks[0]);

    // Display name field
    let name_active = state.editing_field == ParticipantField::DisplayName;
    let name_border = if name_active {
        theme.border_focused
    } else {
        theme.border
    };
    let name_block = Block::default()
        .title(format!(
            " {} {} ",
            ParticipantField::DisplayName.display_name(),
            if name_active { "(editing)" } else { "" }
        ))
        .borders(Borders::ALL)
        .border_style(name_border);

    let name_text = if name_active {
        Paragraph::new(state.input.as_str()).style(theme.assistant)
    } else {
        Paragraph::new(participant.display_name.as_str()).style(theme.muted)
    };

    frame.render_widget(name_block.clone(), chunks[1]);
    let name_inner = name_block.inner(chunks[1]);
    frame.render_widget(name_text, name_inner);

    // System prompt field
    let prompt_active = state.editing_field == ParticipantField::SystemPrompt;
    let prompt_border = if prompt_active {
        theme.border_focused
    } else {
        theme.border
    };
    let prompt_block = Block::default()
        .title(format!(
            " {} {} ",
            ParticipantField::SystemPrompt.display_name(),
            if prompt_active { "(editing)" } else { "" }
        ))
        .borders(Borders::ALL)
        .border_style(prompt_border);

    let prompt_text = if prompt_active {
        if state.input.is_empty() {
            Paragraph::new("Optional: Define behavior for this participant...")
                .style(theme.muted)
                .wrap(Wrap { trim: false })
        } else {
            Paragraph::new(state.input.as_str())
                .style(theme.assistant)
                .wrap(Wrap { trim: false })
        }
    } else {
        let text = participant.system_prompt.as_deref().unwrap_or("(none)");
        Paragraph::new(text)
            .style(theme.muted)
            .wrap(Wrap { trim: false })
    };

    frame.render_widget(prompt_block.clone(), chunks[2]);
    let prompt_inner = prompt_block.inner(chunks[2]);
    frame.render_widget(prompt_text, prompt_inner);
}

fn render_mode_step(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &DialogueBuilderState,
    theme: &Theme,
) {
    let items: Vec<ListItem> = state
        .available_modes
        .iter()
        .enumerate()
        .map(|(i, mode)| {
            let marker = if state.mode == *mode { "[x]" } else { "[ ]" };
            let style = if i == state.selected {
                theme.assistant.add_modifier(Modifier::REVERSED)
            } else {
                theme.assistant
            };
            ListItem::new(format!("  {} {}", marker, mode.display_name())).style(style)
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title(" Select Dialogue Mode ")
            .borders(Borders::ALL)
            .border_style(theme.border),
    );
    frame.render_widget(list, area);
}

fn render_prompt_step(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &DialogueBuilderState,
    theme: &Theme,
) {
    let block = Block::default()
        .title(" Initial Prompt (optional) ")
        .borders(Borders::ALL)
        .border_style(theme.border);

    let prompt_text = if state.input.is_empty() && state.initial_prompt.is_empty() {
        Paragraph::new("Enter the topic or question to start the dialogue...")
            .style(theme.muted)
            .wrap(Wrap { trim: false })
    } else {
        let text = if state.input.is_empty() {
            state.initial_prompt.as_str()
        } else {
            state.input.as_str()
        };
        Paragraph::new(text)
            .style(theme.assistant)
            .wrap(Wrap { trim: false })
    };

    frame.render_widget(block.clone(), area);
    let inner = block.inner(area);
    frame.render_widget(prompt_text, inner);
}

fn render_review_step(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &DialogueBuilderState,
    theme: &Theme,
) {
    let mut lines = Vec::new();

    // Mode
    lines.push(Line::from(vec![
        Span::styled("Mode: ", theme.accent),
        Span::styled(state.mode.display_name(), theme.assistant),
    ]));
    lines.push(Line::from(""));

    // Participants
    lines.push(Line::styled("Participants:", theme.accent));
    for p in &state.participants {
        let (r, g, b) = p.color.rgb();
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(&p.display_name, Style::default().fg(Color::Rgb(r, g, b))),
            Span::styled(format!(" ({}:{})", p.provider_id, p.model_id), theme.muted),
        ]));
        // Show system prompt if set
        if let Some(ref prompt) = p.system_prompt {
            let preview = if prompt.len() > 40 {
                format!("{}...", &prompt[..40])
            } else {
                prompt.clone()
            };
            lines.push(Line::from(vec![
                Span::styled("    System: ", theme.muted),
                Span::styled(preview, theme.assistant),
            ]));
        }
    }
    lines.push(Line::from(""));

    // Initial prompt
    lines.push(Line::styled("Initial Prompt:", theme.accent));
    if state.initial_prompt.is_empty() {
        lines.push(Line::styled("  (none)", theme.muted));
    } else {
        for line in state.initial_prompt.lines().take(3) {
            lines.push(Line::styled(format!("  {}", line), theme.assistant));
        }
    }

    let text = Text::from(lines);
    let block = Block::default()
        .title(" Review ")
        .borders(Borders::ALL)
        .border_style(theme.border);

    let paragraph = Paragraph::new(text).block(block);
    frame.render_widget(paragraph, area);
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, state: &DialogueBuilderState, theme: &Theme) {
    let help_text = match state.step {
        DialogueBuilderStep::Participants => {
            "Enter: Add | e: Configure | Del: Remove | Tab: Next | Esc: Cancel"
        }
        DialogueBuilderStep::ConfigureParticipant => {
            "Tab: Switch Field | Enter: Save & Back | Esc: Cancel"
        }
        DialogueBuilderStep::Mode => "Enter: Select | Tab: Next | Esc: Back",
        DialogueBuilderStep::InitialPrompt => "Enter: Accept | Tab: Next | Esc: Back",
        DialogueBuilderStep::Review => "Enter: Start Dialogue | Esc: Back",
    };

    let footer_text = if let Some(ref err) = state.error {
        Line::from(vec![
            Span::styled("Error: ", theme.error),
            Span::styled(err.as_str(), theme.error),
        ])
    } else {
        Line::styled(help_text, theme.muted)
    };

    let footer = Paragraph::new(footer_text);
    frame.render_widget(footer, area);
}
