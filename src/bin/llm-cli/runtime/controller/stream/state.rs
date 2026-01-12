use crate::conversation::{
    ConversationMessage, MessageKind, MessageRole, MessageState, ToolInvocation,
};
use crate::runtime::AppStatus;

use super::super::AppController;

impl AppController {
    pub(super) fn append_stream_text(
        &mut self,
        message_id: crate::conversation::MessageId,
        delta: &str,
    ) {
        if let Some(conv) = self.state.active_conversation_mut() {
            if let Some(msg) = conv.messages.iter_mut().find(|m| m.id == message_id) {
                msg.update_text(delta);
                msg.state = MessageState::Streaming;
            }
        }
    }

    pub(super) fn add_tool_call(
        &mut self,
        conv_id: crate::conversation::ConversationId,
        id: String,
        name: String,
    ) {
        if let Some(conv) = self.state.active_conversation_mut() {
            if conv.id != conv_id {
                return;
            }
            let invocation = ToolInvocation {
                id,
                name,
                arguments: String::new(),
                partial: true,
            };
            let mut msg =
                ConversationMessage::new(MessageRole::Tool, MessageKind::ToolCall(invocation));
            msg.state = MessageState::Streaming;
            conv.push_message(msg);
        }
    }

    pub(super) fn update_tool_call(
        &mut self,
        conv_id: crate::conversation::ConversationId,
        id: &str,
        delta: &str,
    ) {
        if let Some(conv) = self.state.active_conversation_mut() {
            if conv.id != conv_id {
                return;
            }
            if let Some(msg) = conv
                .messages
                .iter_mut()
                .find(|m| matches!(&m.kind, MessageKind::ToolCall(inv) if inv.id == id))
            {
                if let MessageKind::ToolCall(inv) = &mut msg.kind {
                    inv.arguments.push_str(delta);
                    msg.bump_version();
                }
            }
        }
    }

    pub(super) async fn complete_tool_call(
        &mut self,
        conv_id: crate::conversation::ConversationId,
        invocation: ToolInvocation,
    ) {
        if let Some(conv) = self.state.active_conversation_mut() {
            if conv.id != conv_id {
                return;
            }
            if let Some(msg) = conv
                .messages
                .iter_mut()
                .find(|m| matches!(&m.kind, MessageKind::ToolCall(inv) if inv.id == invocation.id))
            {
                msg.replace_kind(MessageKind::ToolCall(invocation.clone()));
                msg.state = MessageState::Complete;
            } else {
                let mut msg = ConversationMessage::new(
                    MessageRole::Tool,
                    MessageKind::ToolCall(invocation.clone()),
                );
                msg.state = MessageState::Complete;
                conv.push_message(msg);
            }
        }
        self.schedule_tool_execution(invocation).await;
    }

    pub(super) fn increment_pending_tool(&mut self, conv_id: crate::conversation::ConversationId) {
        let entry = self.pending_tool_calls.entry(conv_id).or_insert(0);
        *entry = entry.saturating_add(1);
    }

    pub(super) fn update_usage(
        &mut self,
        message_id: crate::conversation::MessageId,
        usage: llm::chat::Usage,
    ) {
        let completion_tokens = usage.completion_tokens;
        if let Some(conv) = self.state.active_conversation_mut() {
            if let Some(msg) = conv.messages.iter_mut().find(|m| m.id == message_id) {
                msg.metadata.tokens = Some(usage);
                msg.bump_version();
            }
        }
        self.update_status_tokens(completion_tokens);
    }

    pub(super) fn finish_stream(&mut self) {
        if let Some(conv) = self.state.active_conversation_mut() {
            if let Some(msg) = conv
                .messages
                .iter_mut()
                .rev()
                .find(|m| m.role == MessageRole::Assistant)
            {
                msg.state = MessageState::Complete;
            }
        }
        self.record_snapshot();
    }

    pub(super) fn mark_stream_error(
        &mut self,
        message_id: crate::conversation::MessageId,
        error: String,
    ) {
        if let Some(conv) = self.state.active_conversation_mut() {
            if let Some(msg) = conv.messages.iter_mut().find(|m| m.id == message_id) {
                msg.replace_kind(MessageKind::Error(error));
                msg.state = MessageState::Error;
            }
        }
        self.set_status(AppStatus::Error("stream error".to_string()));
        self.record_snapshot();
    }
}
