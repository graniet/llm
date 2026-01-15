use crossterm::event::{KeyCode, KeyEvent};

use super::super::{OverlayAction, OverlayResult};

pub(super) fn handle_search(
    state: &mut crate::runtime::SearchState,
    key: KeyEvent,
) -> OverlayResult {
    match key.code {
        KeyCode::Esc => OverlayResult::close(OverlayAction::Handled),
        KeyCode::Enter => OverlayResult::action(OverlayAction::JumpToSearch),
        KeyCode::Up => {
            state.prev();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Down => {
            state.next();
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(handle_search_query(state, key.code)),
    }
}

fn handle_search_query(state: &mut crate::runtime::SearchState, code: KeyCode) -> OverlayAction {
    match code {
        KeyCode::Backspace => {
            state.pop_query();
            OverlayAction::UpdateSearch
        }
        KeyCode::Char(ch) => {
            state.push_query(ch);
            OverlayAction::UpdateSearch
        }
        _ => OverlayAction::None,
    }
}
