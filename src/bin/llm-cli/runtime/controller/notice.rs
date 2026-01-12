use crate::conversation::{ConversationMessage, MessageKind, MessageRole, MessageState};

use super::AppController;

impl AppController {
    pub fn push_notice(&mut self, text: impl Into<String>) -> bool {
        let Some(conv) = self.state.active_conversation_mut() else {
            return false;
        };
        let mut msg = ConversationMessage::new(MessageRole::Tool, MessageKind::Text(text.into()));
        msg.state = MessageState::Complete;
        conv.push_message(msg);
        self.state.scroll.reset();
        self.record_snapshot();
        true
    }
}
