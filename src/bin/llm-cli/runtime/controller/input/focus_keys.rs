use crossterm::event::{KeyCode, KeyEvent};

use crate::runtime::Focus;

use super::helpers;
use super::AppController;

pub async fn handle_focus_messages(controller: &mut AppController, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Esc => {
            controller.state.focus = Focus::Input;
            controller.state.selected_message = None;
            true
        }
        KeyCode::Up | KeyCode::Char('k') => controller.select_prev_message(),
        KeyCode::Down | KeyCode::Char('j') => controller.select_next_message(),
        KeyCode::PageUp => helpers::page_up(controller),
        KeyCode::PageDown => helpers::page_down(controller),
        KeyCode::Home | KeyCode::Char('g') => controller.select_first_message(),
        KeyCode::End | KeyCode::Char('G') => controller.select_last_message(),
        KeyCode::Enter => {
            if controller.toggle_selected_tool_output() {
                true
            } else {
                controller.open_pager_for_selected()
            }
        }
        KeyCode::Char('p') => controller.open_pager_for_selected(),
        KeyCode::Char('D') => controller.open_diff_for_selected(),
        KeyCode::Char('y') => controller.copy_selected(),
        KeyCode::Char('d') => controller.delete_selected(),
        KeyCode::Char('r') => controller.regenerate_last().await,
        KeyCode::Char('e') => controller.edit_last_user(),
        _ => true,
    }
}
