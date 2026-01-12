use std::time::Instant;

use crate::runtime::OverlayState;

use super::helpers;
use super::AppController;

pub fn handle_char_input(controller: &mut AppController, ch: char) -> bool {
    let now = Instant::now();
    let is_burst = controller.state.paste_detector.record_char(now);
    if !is_burst && controller.state.input.is_empty() {
        match ch {
            '/' => return helpers::open_slash_commands(controller),
            '?' => return helpers::open_help(controller),
            _ => {}
        }
    }
    if matches!(controller.state.overlay, OverlayState::None) {
        controller.state.input.insert_char(ch);
        return true;
    }
    false
}
