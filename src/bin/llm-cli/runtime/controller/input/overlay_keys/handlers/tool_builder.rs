use crossterm::event::{KeyCode, KeyEvent};

use crate::runtime::{ToolBuilderResult, ToolBuilderState};

use super::super::{OverlayAction, OverlayResult};

pub(super) fn handle_tool_builder(state: &mut ToolBuilderState, key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Esc => OverlayResult::close(OverlayAction::None),
        KeyCode::Enter => match state.submit() {
            ToolBuilderResult::Continue => OverlayResult::action(OverlayAction::Handled),
            ToolBuilderResult::Complete(draft) => {
                OverlayResult::close(OverlayAction::ToolBuilderComplete(draft))
            }
            ToolBuilderResult::Cancel => OverlayResult::close(OverlayAction::None),
        },
        KeyCode::Backspace => {
            state.pop_char();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Char(c) => {
            state.push_char(c);
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}
