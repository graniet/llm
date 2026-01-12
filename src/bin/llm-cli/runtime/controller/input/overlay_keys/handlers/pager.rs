use crossterm::event::{KeyCode, KeyEvent};

use crate::runtime::PagerState;

use super::super::{OverlayAction, OverlayResult};
use super::scroll::{pager_height, SCROLL_STEP};

pub(super) fn handle_pager(state: &mut PagerState, key: KeyEvent, height: u16) -> OverlayResult {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => OverlayResult::close(OverlayAction::Handled),
        KeyCode::Up | KeyCode::Char('k') => {
            state.scroll_up(SCROLL_STEP);
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Down | KeyCode::Char('j') => {
            state.scroll_down(SCROLL_STEP, pager_height(height));
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
            state.scroll_top();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::End | KeyCode::Char('G') => {
            state.scroll_bottom(pager_height(height));
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}
