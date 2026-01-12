use crossterm::event::{KeyCode, KeyEvent};

use crate::runtime::InputMode;

use super::char_input::handle_char_input;
use super::helpers;
use super::AppController;

pub async fn handle_vi_key(controller: &mut AppController, key: KeyEvent) -> bool {
    match controller.state.input_mode {
        InputMode::Insert => handle_insert_key(controller, key).await,
        InputMode::Normal => handle_normal_key(controller, key).await,
    }
}

async fn handle_insert_key(controller: &mut AppController, key: KeyEvent) -> bool {
    let width = controller.input_width();
    if handle_insert_mode_switch(controller, &key) {
        return true;
    }
    if handle_insert_navigation(controller, &key, width) {
        return true;
    }
    handle_insert_editing(controller, key)
}

fn handle_insert_mode_switch(controller: &mut AppController, key: &KeyEvent) -> bool {
    if key.code == KeyCode::Esc {
        controller.state.input_mode = InputMode::Normal;
        return true;
    }
    false
}

fn handle_insert_navigation(controller: &mut AppController, key: &KeyEvent, width: u16) -> bool {
    match key.code {
        KeyCode::Up => helpers::move_input_up(controller, width),
        KeyCode::Down => helpers::move_input_down(controller, width),
        KeyCode::Left => {
            controller.state.input.move_left();
            true
        }
        KeyCode::Right => {
            controller.state.input.move_right();
            true
        }
        KeyCode::Home => {
            controller.state.input.move_home();
            true
        }
        KeyCode::End => {
            controller.state.input.move_end();
            true
        }
        _ => false,
    }
}

fn handle_insert_editing(controller: &mut AppController, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Enter => {
            controller.state.input.newline();
            true
        }
        KeyCode::Backspace => {
            controller.state.input.backspace();
            true
        }
        KeyCode::Delete => {
            controller.state.input.delete();
            true
        }
        KeyCode::Char(ch) => handle_char_input(controller, ch),
        _ => false,
    }
}

async fn handle_normal_key(controller: &mut AppController, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('i') => {
            controller.state.input_mode = InputMode::Insert;
            true
        }
        KeyCode::Char('j') | KeyCode::Down => helpers::scroll_down(controller, 1),
        KeyCode::Char('k') | KeyCode::Up => helpers::scroll_up(controller, 1),
        KeyCode::Char('g') => helpers::scroll_to_top(controller),
        KeyCode::Char('G') => helpers::scroll_to_bottom(controller),
        KeyCode::Char('/') => helpers::open_search(controller),
        KeyCode::Char('?') => helpers::open_help(controller),
        KeyCode::Char('r') => controller.regenerate_last().await,
        KeyCode::Char('e') => controller.edit_last_user(),
        KeyCode::Char('y') => controller.copy_selected(),
        KeyCode::Char('d') => controller.delete_selected(),
        _ => false,
    }
}
