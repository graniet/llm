use crossterm::event::{KeyCode, KeyEvent};

use super::super::{OverlayAction, OverlayResult};

pub(super) fn handle_picker(
    state: &mut crate::runtime::PickerState,
    key: KeyEvent,
) -> OverlayResult {
    if key.code == KeyCode::Esc {
        return OverlayResult::close(OverlayAction::Handled);
    }
    if handle_picker_nav(state, key.code) {
        return OverlayResult::action(OverlayAction::Handled);
    }
    if let Some(action) = handle_picker_select(state, key.code) {
        return OverlayResult::close(action);
    }
    handle_picker_query(state, key.code);
    OverlayResult::action(OverlayAction::Handled)
}

pub(super) fn handle_model_picker(
    state: &mut crate::runtime::PickerState,
    key: KeyEvent,
) -> OverlayResult {
    if key.code == KeyCode::Esc {
        return OverlayResult::close(OverlayAction::Handled);
    }
    if handle_picker_nav(state, key.code) {
        return OverlayResult::action(OverlayAction::Handled);
    }
    if let Some(action) = handle_model_select(state, key.code) {
        return OverlayResult::close(action);
    }
    handle_picker_query(state, key.code);
    OverlayResult::action(OverlayAction::Handled)
}

pub(super) fn handle_skill_picker(
    state: &mut crate::runtime::PickerState,
    key: KeyEvent,
) -> OverlayResult {
    if key.code == KeyCode::Esc {
        return OverlayResult::close(OverlayAction::Handled);
    }
    if handle_picker_nav(state, key.code) {
        return OverlayResult::action(OverlayAction::Handled);
    }
    if let Some(action) = handle_skill_select(state, key.code) {
        return OverlayResult::close(action);
    }
    handle_picker_query(state, key.code);
    OverlayResult::action(OverlayAction::Handled)
}

fn handle_picker_nav(state: &mut crate::runtime::PickerState, code: KeyCode) -> bool {
    match code {
        KeyCode::Up => {
            state.prev();
            true
        }
        KeyCode::Down => {
            state.next();
            true
        }
        _ => false,
    }
}

fn handle_picker_select(
    state: &mut crate::runtime::PickerState,
    code: KeyCode,
) -> Option<OverlayAction> {
    if code != KeyCode::Enter {
        return None;
    }
    state
        .selected_item()
        .cloned()
        .map(OverlayAction::PickerSelected)
}

fn handle_picker_query(state: &mut crate::runtime::PickerState, code: KeyCode) {
    match code {
        KeyCode::Backspace => {
            state.pop_query();
        }
        KeyCode::Char(ch) => {
            state.push_query(ch);
        }
        _ => {}
    }
}

fn handle_model_select(
    state: &mut crate::runtime::PickerState,
    code: KeyCode,
) -> Option<OverlayAction> {
    if code != KeyCode::Enter {
        return None;
    }
    if let Some(item) = state.selected_item() {
        return Some(OverlayAction::SetModel(item.id.clone()));
    }
    if !state.query.is_empty() {
        return Some(OverlayAction::SetModel(state.query.clone()));
    }
    Some(OverlayAction::Handled)
}

fn handle_skill_select(
    state: &mut crate::runtime::PickerState,
    code: KeyCode,
) -> Option<OverlayAction> {
    if code != KeyCode::Enter {
        return None;
    }
    state
        .selected_item()
        .map(|item| OverlayAction::ActivateSkill(item.id.clone()))
        .or_else(|| {
            (!state.query.is_empty()).then(|| OverlayAction::ActivateSkill(state.query.clone()))
        })
}
