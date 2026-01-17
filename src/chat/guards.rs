use crate::{chat::ChatMessage, error::LLMError};

pub(crate) fn ensure_no_audio(
    messages: &[ChatMessage],
    error_message: &str,
) -> Result<(), LLMError> {
    if messages.iter().any(ChatMessage::has_audio) {
        return Err(LLMError::InvalidRequest(error_message.to_string()));
    }
    Ok(())
}
