use crossterm::event::{KeyCode, KeyEvent};

use super::super::{OverlayAction, OverlayResult};

pub(super) fn handle_help(key: KeyEvent) -> OverlayResult {
    if key.code == KeyCode::Esc || key.code == KeyCode::Char('?') {
        return OverlayResult::close(OverlayAction::Handled);
    }
    OverlayResult::action(OverlayAction::None)
}
