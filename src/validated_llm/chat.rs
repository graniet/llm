use async_trait::async_trait;

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, Tool},
    error::LLMError,
};

use super::helpers::decrement_attempts;
use super::wrapper::ValidatedLLM;

#[async_trait]
impl ChatProvider for ValidatedLLM {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let mut local_messages = messages.to_vec();
        let mut remaining_attempts = self.attempts();

        loop {
            let response = self.inner().chat_with_tools(&local_messages, tools).await?;
            let text = response.text().unwrap_or_default();

            match (self.validator())(&text) {
                Ok(()) => return Ok(response),
                Err(err) => {
                    remaining_attempts = decrement_attempts(remaining_attempts, &err)?;
                    append_validation_feedback(&mut local_messages, &err);
                }
            }
        }
    }
}

fn append_validation_feedback(messages: &mut Vec<ChatMessage>, err: &str) {
    messages.push(ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: format!(
            "Your previous output was invalid because: {err}\nPlease try again and produce a valid response."
        ),
    });
}
