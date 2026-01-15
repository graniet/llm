mod char_input;
mod focus_keys;
pub(super) mod helpers;
mod main_keys;
mod overlay_keys;
mod simple_keys;
mod slash_overlay;
mod vi_keys;

use std::time::Instant;

use crate::runtime::InputEvent;
use crate::runtime::OverlayState;

use super::AppController;

pub async fn handle_input(controller: &mut AppController, event: InputEvent) -> bool {
    match event {
        InputEvent::Key(key) => main_keys::dispatch_key(controller, key).await,
        InputEvent::Mouse(mouse) => helpers::handle_mouse(controller, mouse),
        InputEvent::Paste(text) => handle_paste_event(controller, text),
        InputEvent::Resize(w, h) => {
            controller.state.terminal_size = (w, h);
            true
        }
    }
}

pub fn handle_tick(controller: &mut AppController) -> bool {
    let mut dirty = controller
        .state
        .animation
        .tick(&controller.state.terminal_caps);
    if controller.state.status_metrics.should_redraw() {
        dirty = true;
    }
    dirty
}

fn handle_paste_event(controller: &mut AppController, text: String) -> bool {
    if let OverlayState::Onboarding(state) = &mut controller.state.overlay {
        state.api_key.push_str(&text);
        controller.state.paste_detector.record_paste(Instant::now());
        return true;
    }
    helpers::handle_paste(controller, text)
}
