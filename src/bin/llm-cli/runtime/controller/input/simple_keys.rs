use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::char_input::handle_char_input;
use super::helpers;
use super::AppController;

pub async fn handle_simple_key(controller: &mut AppController, key: KeyEvent) -> bool {
    if handle_simple_navigation(controller, &key) {
        return true;
    }
    handle_simple_editing(controller, key).await
}

fn handle_simple_navigation(controller: &mut AppController, key: &KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::ALT) {
        return false;
    }
    match key.code {
        KeyCode::Up => helpers::scroll_up(controller, 1),
        KeyCode::Down => helpers::scroll_down(controller, 1),
        KeyCode::PageUp => helpers::page_up(controller),
        KeyCode::PageDown => helpers::page_down(controller),
        _ => false,
    }
}

async fn handle_simple_editing(controller: &mut AppController, key: KeyEvent) -> bool {
    if let Some(result) = handle_simple_enter(controller, &key).await {
        return result;
    }
    let width = controller.input_width();
    if handle_simple_movement(controller, &key, width) {
        return true;
    }
    handle_simple_text(controller, key)
}

async fn handle_simple_enter(controller: &mut AppController, key: &KeyEvent) -> Option<bool> {
    if key.code != KeyCode::Enter {
        return None;
    }
    if key.modifiers.contains(KeyModifiers::SHIFT) {
        controller.state.input.newline();
        return Some(true);
    }
    Some(controller.send_user_message().await)
}

fn handle_simple_movement(controller: &mut AppController, key: &KeyEvent, width: u16) -> bool {
    match key.code {
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
        KeyCode::Up if key.modifiers.contains(KeyModifiers::ALT) => {
            helpers::move_input_up(controller, width)
        }
        KeyCode::Down if key.modifiers.contains(KeyModifiers::ALT) => {
            helpers::move_input_down(controller, width)
        }
        _ => false,
    }
}

fn handle_simple_text(controller: &mut AppController, key: KeyEvent) -> bool {
    match key.code {
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
