use crate::config::AppConfig;
use crate::conversation::{
    to_chat_messages, Conversation, ConversationMessage, MessageKind, MessageRole, MessageState,
};
use chrono::Utc;
use llm::chat::ChatMessage;

const ESTIMATED_CHARS_PER_TOKEN: u32 = 4;
const MIN_TOKEN_COUNT: u32 = 1;
pub(crate) const SUMMARY_SAMPLE_COUNT: usize = 6;
pub(crate) const SUMMARY_TAIL_KEEP: usize = 4;
const SUMMARY_MAX_CHARS: usize = 400;
#[derive(Debug, Clone, Copy)]
pub struct ContextUsage {
    pub used_tokens: u32,
    pub max_tokens: u32,
}
impl ContextUsage {
    pub fn percent(self) -> f32 {
        if self.max_tokens == 0 {
            return 0.0;
        }
        (self.used_tokens as f32 / self.max_tokens as f32) * 100.0
    }
}
pub fn usage_for(conversation: &Conversation, config: &AppConfig) -> ContextUsage {
    let max_tokens = context_limit(conversation, config);
    let used_tokens = estimate_tokens(
        &conversation.messages,
        conversation.system_prompt.as_deref(),
    );
    ContextUsage {
        used_tokens,
        max_tokens,
    }
}
pub fn context_limit(conversation: &Conversation, config: &AppConfig) -> u32 {
    let mut limit = config.chat.max_context_tokens;
    let provider_cfg = config.providers.get(conversation.provider_id.as_str());
    if let (Some(cfg), Some(model)) = (provider_cfg, conversation.model.as_deref()) {
        if let Some(window) = cfg
            .models
            .iter()
            .find(|m| m.id == model)
            .and_then(|m| m.context_window)
        {
            limit = limit.min(window);
        }
    }
    limit
}
pub fn estimate_tokens(messages: &[ConversationMessage], system_prompt: Option<&str>) -> u32 {
    let mut total = estimate_chat_tokens(&to_chat_messages(messages));
    if let Some(prompt) = system_prompt {
        total = total.saturating_add(estimate_text_tokens(prompt));
    }
    total
}
pub(crate) fn estimate_chat_tokens(messages: &[ChatMessage]) -> u32 {
    messages.iter().map(estimate_message_tokens).sum()
}
pub(crate) fn summary_text_from_chat(messages: &[ChatMessage]) -> String {
    let joined = messages
        .iter()
        .take(SUMMARY_SAMPLE_COUNT)
        .map(|msg| msg.content.clone())
        .collect::<Vec<_>>()
        .join(" | ");
    format!("Summary of earlier turns: {}", truncate_summary(&joined))
}
pub(crate) fn summary_text_from_conversation(messages: &[ConversationMessage]) -> String {
    let joined = messages
        .iter()
        .take(SUMMARY_SAMPLE_COUNT)
        .map(conversation_snippet)
        .collect::<Vec<_>>()
        .join(" | ");
    format!("Summary of earlier turns: {}", truncate_summary(&joined))
}
pub fn summarize_conversation_head(conversation: &mut Conversation, count: usize) -> bool {
    if conversation.messages.len() <= count || count == 0 {
        return false;
    }
    let head = conversation.messages[..count].to_vec();
    let summary_text = summary_text_from_conversation(&head);
    replace_with_summary(conversation, summary_text, count);
    true
}
pub fn compact_conversation(conversation: &mut Conversation, max_tokens: u32) -> bool {
    let used = estimate_tokens(
        &conversation.messages,
        conversation.system_prompt.as_deref(),
    );
    if used <= max_tokens {
        return false;
    }
    let summary_text = summary_text_from_conversation(&conversation.messages);
    let keep_start = conversation
        .messages
        .len()
        .saturating_sub(SUMMARY_TAIL_KEEP)
        .min(conversation.messages.len());
    let tail = conversation.messages[keep_start..].to_vec();
    conversation.messages.clear();
    conversation.messages.push(summary_message(summary_text));
    conversation.messages.extend(tail);
    conversation.updated_at = Utc::now();
    conversation.dirty = true;
    true
}
fn estimate_message_tokens(message: &ChatMessage) -> u32 {
    let content_tokens = estimate_text_tokens(&message.content);
    match &message.message_type {
        llm::chat::MessageType::Text => content_tokens,
        llm::chat::MessageType::Audio(_) => content_tokens,
        llm::chat::MessageType::Image(_) => content_tokens,
        llm::chat::MessageType::Pdf(_) => content_tokens,
        llm::chat::MessageType::ImageURL(_) => content_tokens,
        llm::chat::MessageType::ToolUse(calls) => {
            content_tokens
                + calls
                    .iter()
                    .map(|c| estimate_text_tokens(&c.function.arguments))
                    .sum::<u32>()
        }
        llm::chat::MessageType::ToolResult(calls) => {
            content_tokens
                + calls
                    .iter()
                    .map(|c| estimate_text_tokens(&c.function.arguments))
                    .sum::<u32>()
        }
    }
}
pub(crate) fn estimate_text_tokens(text: &str) -> u32 {
    let chars = text.chars().count() as u32;
    (chars / ESTIMATED_CHARS_PER_TOKEN).max(MIN_TOKEN_COUNT)
}
fn truncate_summary(text: &str) -> String {
    let mut out = text.chars().take(SUMMARY_MAX_CHARS).collect::<String>();
    if text.chars().count() > SUMMARY_MAX_CHARS {
        out.push_str("...");
    }
    out
}
fn conversation_snippet(message: &ConversationMessage) -> String {
    match &message.kind {
        crate::conversation::MessageKind::Text(text) => text.clone(),
        crate::conversation::MessageKind::ToolCall(invocation) => invocation.arguments.clone(),
        crate::conversation::MessageKind::ToolResult(result) => result.output.clone(),
        crate::conversation::MessageKind::Error(text) => text.clone(),
    }
}
fn replace_with_summary(conversation: &mut Conversation, summary: String, count: usize) {
    let mut summary_msg = summary_message(summary);
    summary_msg.state = MessageState::Complete;
    conversation.messages.splice(0..count, vec![summary_msg]);
    conversation.updated_at = Utc::now();
    conversation.dirty = true;
}
fn summary_message(summary: String) -> ConversationMessage {
    let mut msg = ConversationMessage::new(MessageRole::Assistant, MessageKind::Text(summary));
    msg.state = MessageState::Complete;
    msg
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ProviderId;

    fn build_conversation(count: usize) -> Conversation {
        let provider = ProviderId::new("openai");
        let mut conv = Conversation::new(provider, None, None);
        for idx in 0..count {
            let text = format!("msg {idx}");
            conv.push_message(ConversationMessage::new(
                MessageRole::User,
                MessageKind::Text(text),
            ));
        }
        conv
    }
    #[test]
    fn summarize_replaces_head() {
        let mut conv = build_conversation(3);
        assert!(summarize_conversation_head(&mut conv, 2));
        assert_eq!(conv.messages.len(), 2);
        let first = match &conv.messages[0].kind {
            MessageKind::Text(text) => text,
            _ => panic!("expected text summary"),
        };
        assert!(first.starts_with("Summary of earlier turns"));
    }

    #[test]
    fn compact_reduces_messages() {
        let mut conv = build_conversation(12);
        assert!(compact_conversation(&mut conv, 1));
        assert!(conv.messages.len() <= SUMMARY_TAIL_KEEP + 1);
    }
}
