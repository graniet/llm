use crate::runtime::Focus;

use super::AppController;

impl AppController {
    pub fn toggle_focus(&mut self) -> bool {
        self.state.focus = match self.state.focus {
            Focus::Input => Focus::Messages,
            Focus::Messages => Focus::Input,
        };
        if self.state.focus == Focus::Messages {
            self.select_last_message();
        } else {
            self.state.selected_message = None;
        }
        true
    }
}
