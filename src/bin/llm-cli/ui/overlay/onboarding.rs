use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::runtime::{OnboardingState, OnboardingStep};

use super::super::theme::Theme;

pub fn render_onboarding(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &OnboardingState,
    theme: &Theme,
) {
    // Clear the area first
    frame.render_widget(Clear, area);

    let title = step_title(state.step);
    let lines = build_lines(state, theme);
    let paragraph = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_focused)
                .style(Style::default().bg(Color::Black))
                .title(format!(" {} ", title)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn step_title(step: OnboardingStep) -> &'static str {
    match step {
        OnboardingStep::Welcome => "Welcome",
        OnboardingStep::Provider => "Choose Provider",
        OnboardingStep::ApiKey => "API Key",
        OnboardingStep::Preferences => "Preferences",
        OnboardingStep::Confirm => "Confirm",
    }
}

fn build_lines(state: &OnboardingState, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines = match state.step {
        OnboardingStep::Welcome => welcome_lines(),
        OnboardingStep::Provider => provider_lines(state),
        OnboardingStep::ApiKey => api_key_lines(state),
        OnboardingStep::Preferences => preference_lines(state),
        OnboardingStep::Confirm => confirm_lines(state),
    };
    if let Some(err) = &state.error {
        lines.push(Line::styled(format!("Error: {err}"), theme.error));
    }
    lines
}

fn welcome_lines() -> Vec<Line<'static>> {
    vec![
        Line::from("LLM CLI"),
        Line::from("----------------"),
        Line::from("Welcome! Let's set things up."),
        Line::from("Press Enter to begin."),
    ]
}

fn provider_lines(state: &OnboardingState) -> Vec<Line<'static>> {
    let mut lines = vec![Line::from("Select a provider (Up/Down, Enter):")];
    for (idx, provider) in state.providers.iter().enumerate() {
        let marker = if idx == state.selected { ">" } else { " " };
        lines.push(Line::from(format!("{marker} {}", provider.name)));
    }
    lines
}

fn api_key_lines(state: &OnboardingState) -> Vec<Line<'static>> {
    let name = state
        .selected_provider()
        .map(|provider| provider.name.clone())
        .unwrap_or_else(|| "provider".to_string());
    let masked = if state.api_key.is_empty() {
        "<empty>".to_string()
    } else {
        "*".repeat(state.api_key.len())
    };
    vec![
        Line::from(format!("Enter API key for {name}:")),
        Line::from(masked),
        Line::from("Press Enter to continue, Esc to go back."),
    ]
}

fn preference_lines(state: &OnboardingState) -> Vec<Line<'static>> {
    vec![
        Line::from("Preferences:"),
        Line::from(format!("Mode: {:?}", state.mode)),
        Line::from(format!("Theme: {}", state.theme)),
        Line::from("Left/Right mode · Up/Down theme · Enter to continue."),
    ]
}

fn confirm_lines(state: &OnboardingState) -> Vec<Line<'static>> {
    let provider = state
        .selected_provider()
        .map(|p| p.id.as_str().to_string())
        .unwrap_or_else(|| "-".to_string());
    vec![
        Line::from("Ready to go!"),
        Line::from(format!("Provider: {provider}")),
        Line::from(format!("Mode: {:?}", state.mode)),
        Line::from(format!("Theme: {}", state.theme)),
        Line::from("Press Enter to finish."),
    ]
}
