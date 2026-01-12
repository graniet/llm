mod messages;
mod search;

use crate::conversation::{ConversationMessage, MessageKind, MessageRole};
use crate::provider::ProviderId;

use super::AppController;

const INPUT_PADDING: u16 = 2;

impl AppController {
    pub fn input_width(&self) -> u16 {
        let total = self.state.terminal_size.0.max(1);
        total.saturating_sub(INPUT_PADDING)
    }

    pub fn try_history_prev(&mut self, width: u16) -> bool {
        let (row, _) = self.state.input.cursor_position(width);
        if row > 0 {
            return false;
        }
        if let Some(text) = self.state.history.previous(self.state.input.text()) {
            self.state.input.set_text(text);
            return true;
        }
        false
    }

    pub fn try_history_next(&mut self, width: u16) -> bool {
        let lines = self.state.input.wrapped_lines(width);
        let (row, _) = self.state.input.cursor_position(width);
        if row + 1 < lines.len() as u16 {
            return false;
        }
        if let Some(text) = self.state.history.next() {
            self.state.input.set_text(text);
            return true;
        }
        false
    }

    pub fn switch_provider(&mut self, provider_id: String) {
        let conv_id = match self.state.active_conversation_mut() {
            Some(conv) => {
                conv.provider_id = ProviderId::new(provider_id);
                conv.model = None;
                conv.id
            }
            None => return,
        };
        self.state.provider_cache.remove(&conv_id);
    }

    pub fn set_model(&mut self, model: String) {
        let conv_id = match self.state.active_conversation_mut() {
            Some(conv) => {
                conv.model = Some(model);
                conv.id
            }
            None => return,
        };
        self.state.provider_cache.remove(&conv_id);
    }

    pub async fn maybe_start_followup(&mut self, conv_id: crate::conversation::ConversationId) {
        if self.pending_tool_calls.contains_key(&conv_id) {
            return;
        }
        if self.state.status.is_busy() {
            return;
        }
        let assistant_id = self.append_placeholder(conv_id);
        if let Some(request) = self.build_stream_request(conv_id, assistant_id).await {
            self.stream_manager.start(request);
            self.set_status(crate::runtime::AppStatus::Streaming);
        }
    }

    pub async fn regenerate_last(&mut self) -> bool {
        let conv_id = match self.state.active_conversation_id() {
            Some(id) => id,
            None => return false,
        };
        if let Some(conv) = self.state.active_conversation_mut() {
            if let Some(idx) = conv
                .messages
                .iter()
                .rposition(|m| m.role == MessageRole::Assistant)
            {
                conv.messages.remove(idx);
                self.record_snapshot();
            }
        }
        let assistant_id = self.append_placeholder(conv_id);
        self.set_status(crate::runtime::AppStatus::Thinking);
        if let Some(request) = self.build_stream_request(conv_id, assistant_id).await {
            self.stream_manager.start(request);
            self.set_status(crate::runtime::AppStatus::Streaming);
            return true;
        }
        self.set_status(crate::runtime::AppStatus::Idle);
        false
    }

    pub fn append_placeholder(
        &mut self,
        conv_id: crate::conversation::ConversationId,
    ) -> crate::conversation::MessageId {
        let mut msg =
            ConversationMessage::new(MessageRole::Assistant, MessageKind::Text(String::new()));
        msg.state = crate::conversation::MessageState::Streaming;
        let id = msg.id;
        if let Some(conv) = self.state.active_conversation_mut() {
            if conv.id == conv_id {
                conv.push_message(msg);
            }
        }
        id
    }
}
