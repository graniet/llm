use crossterm::event::{KeyCode, KeyEvent};

use crate::provider::backend_env_key;
use crate::runtime::{OnboardingState, OnboardingStep};

use super::super::{OverlayAction, OverlayResult};

pub(super) fn handle_onboarding(state: &mut OnboardingState, key: KeyEvent) -> OverlayResult {
    match state.step {
        OnboardingStep::Welcome => handle_onboarding_welcome(state, key),
        OnboardingStep::Provider => handle_onboarding_provider(state, key),
        OnboardingStep::ApiKey => handle_onboarding_key(state, key),
        OnboardingStep::Preferences => handle_onboarding_preferences(state, key),
        OnboardingStep::Confirm => handle_onboarding_confirm(state, key),
    }
}

fn handle_onboarding_welcome(state: &mut OnboardingState, key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Enter => {
            state.next_step();
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}

fn handle_onboarding_provider(state: &mut OnboardingState, key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Esc => {
            state.prev_step();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Up | KeyCode::Char('k') => {
            state.error = None;
            state.select_prev();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Down | KeyCode::Char('j') => {
            state.error = None;
            state.select_next();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Enter => {
            state.error = None;
            advance_onboarding(state);
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}

fn handle_onboarding_key(state: &mut OnboardingState, key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Esc => {
            state.prev_step();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Backspace => {
            state.error = None;
            state.api_key.pop();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Char(ch) => {
            state.error = None;
            state.api_key.push(ch);
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Enter => {
            if requires_key(state) && state.api_key.trim().is_empty() {
                state.set_error("API key required.");
            } else {
                state.next_step();
            }
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}

fn handle_onboarding_preferences(state: &mut OnboardingState, key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Esc => {
            state.prev_step();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Left | KeyCode::Right => {
            state.mode = toggle_mode(state.mode);
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Up | KeyCode::Down => {
            state.theme = toggle_theme(&state.theme);
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Enter => {
            state.next_step();
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}

fn handle_onboarding_confirm(state: &mut OnboardingState, key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Esc => {
            state.prev_step();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Enter => OverlayResult::action(OverlayAction::FinishOnboarding),
        _ => OverlayResult::action(OverlayAction::None),
    }
}

fn advance_onboarding(state: &mut OnboardingState) {
    if state.selected_provider().is_none() {
        state.set_error("Select a provider.");
        return;
    }
    if requires_key(state) {
        state.step = OnboardingStep::ApiKey;
        return;
    }
    state.step = OnboardingStep::Preferences;
}

fn requires_key(state: &OnboardingState) -> bool {
    state
        .selected_provider()
        .and_then(|provider| backend_env_key(&provider.backend))
        .is_some()
}

fn toggle_mode(mode: crate::config::NavigationMode) -> crate::config::NavigationMode {
    match mode {
        crate::config::NavigationMode::Simple => crate::config::NavigationMode::Vi,
        crate::config::NavigationMode::Vi => crate::config::NavigationMode::Simple,
    }
}

fn toggle_theme(current: &str) -> String {
    match current.to_lowercase().as_str() {
        "codex" => "mono".to_string(),
        _ => "codex".to_string(),
    }
}
