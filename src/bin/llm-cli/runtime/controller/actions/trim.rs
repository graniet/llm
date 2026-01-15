use llm::chat::ChatMessage;

use crate::conversation::{to_chat_messages, ConversationMessage};
use crate::runtime::context::{
    estimate_chat_tokens, estimate_text_tokens, summary_text_from_chat, SUMMARY_TAIL_KEEP,
};

pub(super) fn trim_messages(
    messages: &[ConversationMessage],
    system_prompt: Option<&str>,
    max_tokens: u32,
    strategy: crate::config::TrimStrategy,
) -> Vec<ChatMessage> {
    let mut chat_messages = to_chat_messages(messages);
    let mut estimated = estimate_total_tokens(&chat_messages, system_prompt);
    if estimated <= max_tokens {
        return chat_messages;
    }
    match strategy {
        crate::config::TrimStrategy::SlidingWindow => {
            while estimated > max_tokens && chat_messages.len() > 1 {
                chat_messages.remove(0);
                estimated = estimate_total_tokens(&chat_messages, system_prompt);
            }
            chat_messages
        }
        crate::config::TrimStrategy::Summarize => {
            summarize_messages(chat_messages, system_prompt, max_tokens)
        }
    }
}

fn summarize_messages(
    mut messages: Vec<ChatMessage>,
    system_prompt: Option<&str>,
    max_tokens: u32,
) -> Vec<ChatMessage> {
    if messages.len() <= 2 {
        return messages;
    }
    let summary_text = summary_text_from_chat(&messages);
    messages.drain(0..messages.len().saturating_sub(SUMMARY_TAIL_KEEP));
    let summary_msg = ChatMessage::assistant().content(summary_text).build();
    messages.insert(0, summary_msg);
    while estimate_total_tokens(&messages, system_prompt) > max_tokens && messages.len() > 2 {
        messages.remove(1);
    }
    messages
}

fn estimate_total_tokens(messages: &[ChatMessage], system_prompt: Option<&str>) -> u32 {
    let mut total = estimate_chat_tokens(messages);
    if let Some(prompt) = system_prompt {
        total = total.saturating_add(estimate_text_tokens(prompt));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{MessageKind, MessageRole};
    use proptest::prelude::*;

    fn conv_message(text: &str) -> ConversationMessage {
        ConversationMessage::new(MessageRole::User, MessageKind::Text(text.to_string()))
    }

    fn contents(messages: &[ChatMessage]) -> Vec<String> {
        messages.iter().map(|msg| msg.content.clone()).collect()
    }

    fn is_suffix(original: &[String], trimmed: &[String]) -> bool {
        if trimmed.len() > original.len() {
            return false;
        }
        let start = original.len().saturating_sub(trimmed.len());
        original[start..] == *trimmed
    }

    proptest! {
        #[test]
        fn sliding_window_trims_from_front(
            texts in prop::collection::vec(".{1,40}", 1..20),
            max_tokens in 1u32..60,
        ) {
            let messages: Vec<ConversationMessage> = texts.iter().map(|t| conv_message(t)).collect();
            let trimmed = trim_messages(
                &messages,
                None,
                max_tokens,
                crate::config::TrimStrategy::SlidingWindow,
            );
            let original_contents = contents(&to_chat_messages(&messages));
            let trimmed_contents = contents(&trimmed);
            prop_assert!(is_suffix(&original_contents, &trimmed_contents));
        }
    }
}
