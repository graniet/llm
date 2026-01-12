mod trim;

use crate::conversation::{ConversationMessage, MessageKind, MessageRole};
use crate::provider::{ProviderFactory, ProviderSelection};
use crate::runtime::streaming::StreamRequest;
use crate::runtime::AppStatus;

use super::AppController;
use trim::trim_messages;

impl AppController {
    pub async fn send_user_message(&mut self) -> bool {
        let mut content = self.state.input.take_text();
        if let Some(handled) = self.handle_slash_input(&content).await {
            return handled;
        }
        if self.apply_skill_mention(&mut content) {
            return true;
        }
        let Some((conv_id, user_message)) = self.build_user_message(content) else {
            return false;
        };
        let assistant_id = self.queue_user_message(conv_id, user_message);
        self.start_stream_for(conv_id, assistant_id).await
    }

    pub fn cancel_active_stream(&mut self) {
        if let Some(id) = self.state.active_conversation_id() {
            self.stream_manager.cancel(id);
            self.set_status(AppStatus::Idle);
        }
    }

    fn append_user_message(
        &mut self,
        user_message: ConversationMessage,
        conv_id: crate::conversation::ConversationId,
    ) {
        if let Some(conv) = self.state.conversations.active_mut() {
            if conv.id != conv_id {
                return;
            }
            conv.push_message(user_message);
            conv.title_from_first_user();
            conv.updated_at = chrono::Utc::now();
        }
        self.state.scroll.reset();
        self.record_snapshot();
    }

    fn apply_skill_mention(&mut self, content: &mut String) -> bool {
        let Some((skill, cleaned)) = self.extract_skill_mention(content) else {
            return false;
        };
        let skill = skill.clone();
        *content = cleaned;
        self.activate_skill(&skill);
        content.trim().is_empty()
    }

    fn build_user_message(
        &mut self,
        content: String,
    ) -> Option<(crate::conversation::ConversationId, ConversationMessage)> {
        if content.trim().is_empty() {
            return None;
        }
        let conv_id = self.state.active_conversation_id()?;
        let user_message = ConversationMessage::new(MessageRole::User, MessageKind::Text(content));
        if let MessageKind::Text(text) = &user_message.kind {
            self.state.history.record(text.clone());
        }
        self.state.mark_dirty();
        self.set_status(AppStatus::Thinking);
        Some((conv_id, user_message))
    }

    fn queue_user_message(
        &mut self,
        conv_id: crate::conversation::ConversationId,
        user_message: ConversationMessage,
    ) -> crate::conversation::MessageId {
        self.append_user_message(user_message, conv_id);
        self.maybe_auto_compact();
        self.append_placeholder(conv_id)
    }

    async fn start_stream_for(
        &mut self,
        conv_id: crate::conversation::ConversationId,
        assistant_id: crate::conversation::MessageId,
    ) -> bool {
        if let Some(request) = self.build_stream_request(conv_id, assistant_id).await {
            self.stream_manager.start(request);
            self.set_status(AppStatus::Streaming);
            return true;
        }
        self.set_status(AppStatus::Idle);
        false
    }

    pub(super) async fn build_stream_request(
        &mut self,
        conv_id: crate::conversation::ConversationId,
        message_id: crate::conversation::MessageId,
    ) -> Option<StreamRequest> {
        let conversation = self.state.active_conversation()?.clone();
        let max_tokens = crate::runtime::context_limit(&conversation, &self.state.config);
        let messages = trim_messages(
            &conversation.messages,
            conversation.system_prompt.as_deref(),
            max_tokens,
            self.state.config.chat.trim_strategy,
        );
        let provider = self.ensure_provider_handle(&conversation)?;
        Some(StreamRequest {
            conversation_id: conv_id,
            message_id,
            provider: provider.provider.clone(),
            messages,
            capabilities: provider.capabilities,
        })
    }

    pub(super) fn ensure_provider_handle(
        &mut self,
        conversation: &crate::conversation::Conversation,
    ) -> Option<crate::provider::ProviderHandle> {
        if let Some(handle) = self.state.provider_cache.get(&conversation.id) {
            return Some(handle.clone());
        }
        let selection = ProviderSelection {
            provider_id: conversation.provider_id.clone(),
            model: conversation.model.clone(),
        };
        let mut overrides = self
            .state
            .session_overrides
            .with_tools(self.tool_registry.function_builders());
        overrides.model = conversation.model.clone().or(overrides.model);
        overrides.system = conversation.system_prompt.clone().or(overrides.system);
        let factory = ProviderFactory::new(&self.state.config, &self.state.registry);
        let handle = factory.build(&selection, overrides).ok()?;
        self.state
            .provider_cache
            .insert(conversation.id, handle.clone());
        Some(handle)
    }
}
