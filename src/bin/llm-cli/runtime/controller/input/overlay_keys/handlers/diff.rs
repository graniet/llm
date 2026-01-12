use crossterm::event::{KeyCode, KeyEvent};

use crate::runtime::DiffViewerState;

use super::super::{OverlayAction, OverlayResult};
use super::scroll::pager_height;

pub(super) fn handle_diff_viewer(
    state: &mut DiffViewerState,
    key: KeyEvent,
    height: u16,
) -> OverlayResult {
    if let Some(result) = handle_diff_decision(state, key) {
        return result;
    }
    handle_diff_navigation(state, key, height)
}

fn handle_diff_decision(state: &mut DiffViewerState, key: KeyEvent) -> Option<OverlayResult> {
    let result = match key.code {
        KeyCode::Esc | KeyCode::Char('q') => OverlayResult::close(OverlayAction::Handled),
        KeyCode::Char('y') | KeyCode::Char('a') => {
            state.accept_current();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Char('n') | KeyCode::Char('r') => {
            state.reject_current();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Char('s') => {
            state.skip_current();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Char('A') => {
            state.accept_all();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Enter => OverlayResult::close(OverlayAction::ApplyDiff(state.diff.clone())),
        _ => return None,
    };
    Some(result)
}

fn handle_diff_navigation(
    state: &mut DiffViewerState,
    key: KeyEvent,
    height: u16,
) -> OverlayResult {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            state.prev_hunk();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Down | KeyCode::Char('j') => {
            state.next_hunk();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::PageUp => {
            state.scroll_up(pager_height(height));
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::PageDown => {
            let view_height = pager_height(height);
            state.scroll_down(view_height, view_height);
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Home | KeyCode::Char('g') => {
            state.scroll = 0;
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::End | KeyCode::Char('G') => {
            state.scroll = state.max_scroll(pager_height(height));
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}
