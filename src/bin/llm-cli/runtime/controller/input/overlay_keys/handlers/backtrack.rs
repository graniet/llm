use crossterm::event::{KeyCode, KeyEvent};

use crate::runtime::BacktrackOverlayState;

use super::super::{OverlayAction, OverlayResult};

pub(super) fn handle_backtrack(state: &mut BacktrackOverlayState, key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => OverlayResult::close(OverlayAction::Handled),
        KeyCode::Up | KeyCode::Char('k') => {
            state.prev();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Down | KeyCode::Char('j') => {
            state.next();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Enter => state
            .selected_entry()
            .map(|entry| OverlayResult::close(OverlayAction::RestoreSnapshot(entry.id)))
            .unwrap_or_else(|| OverlayResult::action(OverlayAction::Handled)),
        _ => OverlayResult::action(OverlayAction::None),
    }
}
