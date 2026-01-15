use std::time::Instant;

use crossterm::event::{MouseEvent, MouseEventKind};

use crate::config::NavigationMode;
use crate::runtime::InputMode;

use super::super::AppController;
use super::scrolling::{scroll_down, scroll_up};

const WHEEL_SCROLL_LINES: u16 = 3;

pub fn handle_mouse(controller: &mut AppController, mouse: MouseEvent) -> bool {
    match mouse.kind {
        MouseEventKind::ScrollUp => scroll_up(controller, WHEEL_SCROLL_LINES),
        MouseEventKind::ScrollDown => scroll_down(controller, WHEEL_SCROLL_LINES),
        _ => false,
    }
}

pub fn handle_paste(controller: &mut AppController, text: String) -> bool {
    let mode = controller.state.config.ui.navigation_mode;
    let can_insert = match mode {
        NavigationMode::Simple => true,
        NavigationMode::Vi => controller.state.input_mode == InputMode::Insert,
    };
    if !can_insert {
        return false;
    }
    controller.state.input.insert_str(&text);
    controller.state.paste_detector.record_paste(Instant::now());
    true
}

pub fn move_input_up(controller: &mut AppController, width: u16) -> bool {
    if controller.try_history_prev(width) {
        return true;
    }
    controller.state.input.move_up(width);
    true
}

pub fn move_input_down(controller: &mut AppController, width: u16) -> bool {
    if controller.try_history_next(width) {
        return true;
    }
    controller.state.input.move_down(width);
    true
}
