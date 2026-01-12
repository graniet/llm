use crate::runtime::OverlayState;

use super::super::AppController;
use super::messages::message_text;

impl AppController {
    pub fn update_search_matches(&mut self) {
        let query = match &self.state.overlay {
            OverlayState::Search(state) => state.query.to_lowercase(),
            _ => return,
        };
        let mut matches = Vec::new();
        if let Some(conv) = self.state.active_conversation() {
            for msg in &conv.messages {
                if message_text(msg).to_lowercase().contains(&query) {
                    matches.push(msg.id);
                }
            }
        }
        if let OverlayState::Search(state) = &mut self.state.overlay {
            state.matches = matches;
            state.selected = 0;
        }
    }

    pub fn jump_to_search_match(&mut self) {
        if let OverlayState::Search(state) = &self.state.overlay {
            if let Some(id) = state.matches.get(state.selected).copied() {
                self.state.selected_message = Some(id);
            }
        }
        self.state.overlay = OverlayState::None;
    }
}
