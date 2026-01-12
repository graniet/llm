use crossterm::event::{KeyCode, KeyEvent};

use super::super::{OverlayAction, OverlayResult};

pub(super) fn handle_tool_approval(key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Char('y') => OverlayResult::close(OverlayAction::ToolApproval(true)),
        KeyCode::Char('n') | KeyCode::Esc => {
            OverlayResult::close(OverlayAction::ToolApproval(false))
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}
