use crossterm::event::{KeyCode, KeyEvent};

use super::super::{OverlayAction, OverlayResult};

pub(super) fn handle_confirm(key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Char('y') => OverlayResult::close(OverlayAction::Quit),
        KeyCode::Char('n') | KeyCode::Esc => OverlayResult::close(OverlayAction::Handled),
        _ => OverlayResult::action(OverlayAction::None),
    }
}
