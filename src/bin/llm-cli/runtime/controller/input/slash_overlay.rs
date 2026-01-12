use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent};

use crate::runtime::{OverlayState, SlashCommandId};

use super::AppController;

pub async fn handle_slash_overlay(controller: &mut AppController, key: KeyEvent) -> bool {
    let action = {
        let state = &mut controller.state;
        let (overlay, paste_detector) = (&mut state.overlay, &mut state.paste_detector);
        match overlay {
            OverlayState::SlashCommands(state) => handle_key(state, paste_detector, key),
            _ => return false,
        }
    };
    apply_action(controller, action).await
}

fn handle_key(
    state: &mut crate::runtime::SlashCommandState,
    paste_detector: &mut crate::runtime::PasteDetector,
    key: KeyEvent,
) -> SlashAction {
    match key.code {
        KeyCode::Esc => SlashAction::Close,
        KeyCode::Up => {
            state.prev();
            SlashAction::None
        }
        KeyCode::Down => {
            state.next();
            SlashAction::None
        }
        KeyCode::Enter => {
            if let Some(cmd) = state.selected_command() {
                // If command requires an argument, insert into input instead of executing
                if cmd.arg_hint.is_some() {
                    SlashAction::InsertCommand(format!("/{} ", cmd.name))
                } else {
                    SlashAction::Select(Some(cmd.id))
                }
            } else {
                SlashAction::Select(None)
            }
        }
        KeyCode::Backspace => handle_backspace(state),
        KeyCode::Char(ch) => handle_char(state, paste_detector, ch),
        _ => SlashAction::None,
    }
}

fn handle_backspace(state: &mut crate::runtime::SlashCommandState) -> SlashAction {
    if state.query.is_empty() {
        return SlashAction::Close;
    }
    state.pop_query();
    SlashAction::None
}

fn handle_char(
    state: &mut crate::runtime::SlashCommandState,
    paste_detector: &mut crate::runtime::PasteDetector,
    ch: char,
) -> SlashAction {
    let now = Instant::now();
    if paste_detector.record_char(now) {
        let pending = state.query.clone();
        return SlashAction::PasteAbort(format!("/{pending}{ch}"));
    }
    state.push_query(ch);
    SlashAction::None
}

async fn apply_action(controller: &mut AppController, action: SlashAction) -> bool {
    match action {
        SlashAction::None => true,
        SlashAction::Close => {
            controller.state.overlay = OverlayState::None;
            true
        }
        SlashAction::PasteAbort(text) => {
            controller.state.overlay = OverlayState::None;
            controller.state.input.insert_str(&text);
            true
        }
        SlashAction::Select(Some(command)) => {
            controller.state.overlay = OverlayState::None;
            controller.handle_slash_command(command).await
        }
        SlashAction::Select(None) => {
            controller.state.overlay = OverlayState::None;
            true
        }
        SlashAction::InsertCommand(text) => {
            controller.state.overlay = OverlayState::None;
            controller.state.input.insert_str(&text);
            true
        }
    }
}

enum SlashAction {
    None,
    Close,
    PasteAbort(String),
    Select(Option<SlashCommandId>),
    /// Insert command text into input (for commands that require arguments)
    InsertCommand(String),
}
