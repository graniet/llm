use std::fmt;

use crate::ToolCall;

/// Role of a participant in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    /// The user/human participant in the conversation
    User,
    /// The AI assistant participant in the conversation
    Assistant,
}

/// The supported MIME type of an image.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageMime {
    /// JPEG image
    JPEG,
    /// PNG image
    PNG,
    /// GIF image
    GIF,
    /// WebP image
    WEBP,
}

impl ImageMime {
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImageMime::JPEG => "image/jpeg",
            ImageMime::PNG => "image/png",
            ImageMime::GIF => "image/gif",
            ImageMime::WEBP => "image/webp",
        }
    }
}

/// The type of a message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum MessageType {
    /// A text message
    #[default]
    Text,
    /// An image message
    Image((ImageMime, Vec<u8>)),
    /// PDF message
    Pdf(Vec<u8>),
    /// Audio message for speech-to-text processing
    Audio(Vec<u8>),
    /// An image URL message
    ImageURL(String),
    /// A tool use
    ToolUse(Vec<ToolCall>),
    /// Tool result
    ToolResult(Vec<ToolCall>),
}

/// The type of reasoning effort for a message in a chat conversation.
pub enum ReasoningEffort {
    /// Low reasoning effort
    Low,
    /// Medium reasoning effort
    Medium,
    /// High reasoning effort
    High,
}

impl fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
        };
        write!(f, "{name}")
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// The role of who sent this message (user or assistant)
    pub role: ChatRole,
    /// The type of the message (text, image, audio, video, etc)
    pub message_type: MessageType,
    /// The text content of the message
    pub content: String,
}

impl ChatMessage {
    /// Create a new builder for a user message
    pub fn user() -> ChatMessageBuilder {
        ChatMessageBuilder::new(ChatRole::User)
    }

    /// Create a new builder for an assistant message
    pub fn assistant() -> ChatMessageBuilder {
        ChatMessageBuilder::new(ChatRole::Assistant)
    }

    /// Check whether this message contains audio data.
    pub fn has_audio(&self) -> bool {
        matches!(self.message_type, MessageType::Audio(_))
    }

    /// Get audio bytes if this is an audio message.
    pub fn audio_data(&self) -> Option<&[u8]> {
        match &self.message_type {
            MessageType::Audio(data) => Some(data),
            _ => None,
        }
    }
}

/// Builder for ChatMessage
#[derive(Debug)]
pub struct ChatMessageBuilder {
    role: ChatRole,
    message_type: MessageType,
    content: String,
}

impl ChatMessageBuilder {
    /// Create a new ChatMessageBuilder with specified role
    pub fn new(role: ChatRole) -> Self {
        Self {
            role,
            message_type: MessageType::default(),
            content: String::new(),
        }
    }

    /// Set the message content
    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self
    }

    /// Set the message type as Image
    pub fn image(mut self, image_mime: ImageMime, raw_bytes: Vec<u8>) -> Self {
        self.message_type = MessageType::Image((image_mime, raw_bytes));
        self
    }

    /// Set the message type as Pdf
    pub fn pdf(mut self, raw_bytes: Vec<u8>) -> Self {
        self.message_type = MessageType::Pdf(raw_bytes);
        self
    }

    /// Set the message type as Audio
    pub fn audio(mut self, audio_data: Vec<u8>) -> Self {
        self.message_type = MessageType::Audio(audio_data);
        self
    }

    /// Set the message type as ImageURL
    pub fn image_url(mut self, url: impl Into<String>) -> Self {
        self.message_type = MessageType::ImageURL(url.into());
        self
    }

    /// Set the message type as ToolUse
    pub fn tool_use(mut self, tools: Vec<ToolCall>) -> Self {
        self.message_type = MessageType::ToolUse(tools);
        self
    }

    /// Set the message type as ToolResult
    pub fn tool_result(mut self, tools: Vec<ToolCall>) -> Self {
        self.message_type = MessageType::ToolResult(tools);
        self
    }

    /// Build the ChatMessage
    pub fn build(self) -> ChatMessage {
        ChatMessage {
            role: self.role,
            message_type: self.message_type,
            content: self.content,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_builder_sets_audio_type() {
        let msg = ChatMessage::user().audio(vec![1, 2, 3]).build();
        assert!(msg.has_audio());
        assert_eq!(msg.audio_data(), Some([1, 2, 3].as_slice()));
    }

    #[test]
    fn audio_accessors_ignore_non_audio() {
        let msg = ChatMessage::assistant().content("hi").build();
        assert!(!msg.has_audio());
        assert_eq!(msg.audio_data(), None);
    }
}
