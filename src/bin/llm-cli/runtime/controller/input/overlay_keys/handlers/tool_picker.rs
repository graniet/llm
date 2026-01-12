use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::runtime::PickerState;

use super::super::{OverlayAction, OverlayResult};

pub(super) fn handle_tool_picker(state: &mut PickerState, key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Esc => OverlayResult::close(OverlayAction::None),
        KeyCode::Enter => {
            if let Some(item) = state.selected_item() {
                OverlayResult::close(OverlayAction::ToolSelected(item.id.clone()))
            } else {
                OverlayResult::close(OverlayAction::None)
            }
        }
        // 'd' or 'x' to delete/remove a tool
        KeyCode::Char('d') | KeyCode::Char('x') => {
            if let Some(item) = state.selected_item() {
                // Only allow removing custom tools (not builtin)
                if item.badges.iter().any(|b| b == "custom") {
                    OverlayResult::close(OverlayAction::ToolRemove(item.id.clone()))
                } else {
                    OverlayResult::action(OverlayAction::None)
                }
            } else {
                OverlayResult::action(OverlayAction::None)
            }
        }
        KeyCode::Up | KeyCode::Char('k') => {
            state.prev();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Down | KeyCode::Char('j') => {
            state.next();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Char(c) if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT => {
            state.push_query(c);
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Backspace => {
            state.pop_query();
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}
