use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{error::LLMError, ToolCall};

/// Usage metadata for a chat response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    #[serde(alias = "input_tokens")]
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    #[serde(alias = "output_tokens")]
    pub completion_tokens: u32,
    /// Total number of tokens used
    pub total_tokens: u32,
    /// Breakdown of completion tokens, if available
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "output_tokens_details"
    )]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Breakdown of prompt tokens, if available
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "input_tokens_details"
    )]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Stream response chunk that mimics OpenAI's streaming response format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResponse {
    /// Array of choices in the response
    pub choices: Vec<StreamChoice>,
    /// Usage metadata, typically present in the final chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Individual choice in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    /// Delta containing the incremental content
    pub delta: StreamDelta,
}

/// Delta content in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDelta {
    /// The incremental content, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The incremental tool calls, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Breakdown of completion tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    /// Tokens used for reasoning (for reasoning models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    /// Tokens used for audio output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

/// Breakdown of prompt tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    /// Tokens used for cached content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    /// Tokens used for audio input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

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

/// Represents a parameter in a function tool
#[derive(Debug, Clone, Serialize)]
pub struct ParameterProperty {
    /// The type of the parameter (e.g. "string", "number", "array", etc)
    #[serde(rename = "type")]
    pub property_type: String,
    /// Description of what the parameter does
    pub description: String,
    /// When type is "array", this defines the type of the array items
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<ParameterProperty>>,
    /// When type is "enum", this defines the possible values for the parameter
    #[serde(skip_serializing_if = "Option::is_none", rename = "enum")]
    pub enum_list: Option<Vec<String>>,
}

/// Represents the parameters schema for a function tool
#[derive(Debug, Clone, Serialize)]
pub struct ParametersSchema {
    /// The type of the parameters object (usually "object")
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Map of parameter names to their properties
    pub properties: HashMap<String, ParameterProperty>,
    /// List of required parameter names
    pub required: Vec<String>,
}

/// Represents a function definition for a tool.
///
/// The `parameters` field stores the JSON Schema describing the function
/// arguments.  It is kept as a raw `serde_json::Value` to allow arbitrary
/// complexity (nested arrays/objects, `oneOf`, etc.) without requiring a
/// bespoke Rust structure.
///
/// Builder helpers can still generate simple schemas automatically, but the
/// user may also provide any valid schema directly.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionTool {
    /// Name of the function
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// JSON Schema describing the parameters
    pub parameters: Value,
}

/// Defines rules for structured output responses based on [OpenAI's structured output requirements](https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format).
/// Individual providers may have additional requirements or restrictions, but these should be handled by each provider's backend implementation.
///
/// If you plan on deserializing into this struct, make sure the source text has a `"name"` field, since that's technically the only thing required by OpenAI.
///
/// ## Example
///
/// ```
/// use llm::chat::StructuredOutputFormat;
/// use serde_json::json;
///
/// let response_format = r#"
///     {
///         "name": "Student",
///         "description": "A student object",
///         "schema": {
///             "type": "object",
///             "properties": {
///                 "name": {
///                     "type": "string"
///                 },
///                 "age": {
///                     "type": "integer"
///                 },
///                 "is_student": {
///                     "type": "boolean"
///                 }
///             },
///             "required": ["name", "age", "is_student"]
///         }
///     }
/// "#;
/// let structured_output: StructuredOutputFormat = serde_json::from_str(response_format).unwrap();
/// assert_eq!(structured_output.name, "Student");
/// assert_eq!(structured_output.description, Some("A student object".to_string()));
/// ```
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]

pub struct StructuredOutputFormat {
    /// Name of the schema
    pub name: String,
    /// The description of the schema
    pub description: Option<String>,
    /// The JSON schema for the structured output
    pub schema: Option<Value>,
    /// Whether to enable strict schema adherence
    pub strict: Option<bool>,
}

/// Represents a tool that can be used in chat
#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    /// The type of tool (e.g. "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition if this is a function tool
    pub function: FunctionTool,
}

/// Tool choice determines how the LLM uses available tools.
/// The behavior is standardized across different LLM providers.
#[derive(Debug, Clone, Default)]
pub enum ToolChoice {
    /// Model can use any tool, but it must use at least one.
    /// This is useful when you want to force the model to use tools.
    Any,

    /// Model can use any tool, and may elect to use none.
    /// This is the default behavior and gives the model flexibility.
    #[default]
    Auto,

    /// Model must use the specified tool and only the specified tool.
    /// The string parameter is the name of the required tool.
    /// This is useful when you want the model to call a specific function.
    Tool(String),

    /// Explicitly disables the use of tools.
    /// The model will not use any tools even if they are provided.
    None,
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ToolChoice::Any => serializer.serialize_str("required"),
            ToolChoice::Auto => serializer.serialize_str("auto"),
            ToolChoice::None => serializer.serialize_str("none"),
            ToolChoice::Tool(name) => {
                use serde::ser::SerializeMap;

                // For tool_choice: {"type": "function", "function": {"name": "function_name"}}
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "function")?;

                // Inner function object
                let mut function_obj = std::collections::HashMap::new();
                function_obj.insert("name", name.as_str());

                map.serialize_entry("function", &function_obj)?;
                map.end()
            }
        }
    }
}

pub trait ChatResponse: std::fmt::Debug + std::fmt::Display + Send + Sync {
    fn text(&self) -> Option<String>;
    fn tool_calls(&self) -> Option<Vec<ToolCall>>;
    fn thinking(&self) -> Option<String> {
        None
    }
    fn usage(&self) -> Option<Usage> {
        None
    }
}

/// Trait for providers that support chat-style interactions.
#[async_trait]
pub trait ChatProvider: Sync + Send {
    /// Sends a chat request to the provider with a sequence of messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a chat request to the provider with a sequence of messages and tools.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    /// * `tools` - Optional slice of tools to use in the chat
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError>;

    /// Sends a chat with web search request to the provider
    ///
    /// # Arguments
    ///
    /// * `input` - The input message
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat_with_web_search(
        &self,
        _input: String,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Err(LLMError::Generic(
            "Web search not supported for this provider".to_string(),
        ))
    }

    /// Sends a streaming chat request to the provider with a sequence of messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// A stream of text tokens or an error
    async fn chat_stream(
        &self,
        _messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        Err(LLMError::Generic(
            "Streaming not supported for this provider".to_string(),
        ))
    }

    /// Sends a streaming chat request that returns structured response chunks.
    ///
    /// ⚠️ Getting usage metadata while streaming have been noticed to be a unstable depending on the provider
    /// (it can be missing).
    ///
    /// This method returns a stream of `StreamResponse` objects that mimic OpenAI's
    /// streaming response format with `.choices[0].delta.content` and `.usage`.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// A stream of `StreamResponse` objects or an error
    async fn chat_stream_struct(
        &self,
        _messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        LLMError,
    > {
        Err(LLMError::Generic(
            "Structured streaming not supported for this provider".to_string(),
        ))
    }

    /// Get current memory contents if provider supports memory
    async fn memory_contents(&self) -> Option<Vec<ChatMessage>> {
        None
    }

    /// Summarizes a conversation history into a concise 2-3 sentence summary
    ///
    /// # Arguments
    /// * `msgs` - The conversation messages to summarize
    ///
    /// # Returns
    /// A string containing the summary or an error if summarization fails
    async fn summarize_history(&self, msgs: &[ChatMessage]) -> Result<String, LLMError> {
        let prompt = format!(
            "Summarize in 2-3 sentences:\n{}",
            msgs.iter()
                .map(|m| format!("{:?}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let req = [ChatMessage::user().content(prompt).build()];
        self.chat(&req)
            .await?
            .text()
            .ok_or(LLMError::Generic("no text in summary response".into()))
    }
}

impl fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReasoningEffort::Low => write!(f, "low"),
            ReasoningEffort::Medium => write!(f, "medium"),
            ReasoningEffort::High => write!(f, "high"),
        }
    }
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
    pub fn content<S: Into<String>>(mut self, content: S) -> Self {
        self.content = content.into();
        self
    }

    /// Set the message type as Image
    pub fn image(mut self, image_mime: ImageMime, raw_bytes: Vec<u8>) -> Self {
        self.message_type = MessageType::Image((image_mime, raw_bytes));
        self
    }

    /// Set the message type as Image
    pub fn pdf(mut self, raw_bytes: Vec<u8>) -> Self {
        self.message_type = MessageType::Pdf(raw_bytes);
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

/// Creates a Server-Sent Events (SSE) stream from an HTTP response.
///
/// # Arguments
///
/// * `response` - The HTTP response from the streaming API
/// * `parser` - Function to parse each SSE chunk into optional text content
///
/// # Returns
///
/// A pinned stream of text tokens or an error
pub(crate) fn create_sse_stream<F>(
    response: reqwest::Response,
    parser: F,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>
where
    F: Fn(&str) -> Result<Option<String>, LLMError> + Send + 'static,
{
    let stream = response
        .bytes_stream()
        .scan(
            (String::new(), Vec::new()),
            move |(buffer, utf8_buffer), chunk| {
                let result = match chunk {
                    Ok(bytes) => {
                        utf8_buffer.extend_from_slice(&bytes);

                        match String::from_utf8(utf8_buffer.clone()) {
                            Ok(text) => {
                                buffer.push_str(&text);
                                utf8_buffer.clear();
                            }
                            Err(e) => {
                                let valid_up_to = e.utf8_error().valid_up_to();
                                if valid_up_to > 0 {
                                    // Safe to use from_utf8_lossy here since valid_up_to points to
                                    // a valid UTF-8 boundary - no replacement characters will be introduced
                                    let valid =
                                        String::from_utf8_lossy(&utf8_buffer[..valid_up_to]);
                                    buffer.push_str(&valid);
                                    utf8_buffer.drain(..valid_up_to);
                                }
                            }
                        }

                        let mut results = Vec::new();

                        while let Some(pos) = buffer.find("\n\n") {
                            let event = buffer[..pos + 2].to_string();
                            buffer.drain(..pos + 2);

                            match parser(&event) {
                                Ok(Some(content)) => results.push(Ok(content)),
                                Ok(None) => {}
                                Err(e) => results.push(Err(e)),
                            }
                        }

                        Some(results)
                    }
                    Err(e) => Some(vec![Err(LLMError::HttpError(e.to_string()))]),
                };

                async move { result }
            },
        )
        .flat_map(futures::stream::iter);

    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream::StreamExt;

    #[tokio::test]
    async fn test_create_sse_stream_handles_split_utf8() {
        let test_data = "data: Positive reactions\n\n".as_bytes();

        let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![
            Ok(Bytes::from(&test_data[..10])),
            Ok(Bytes::from(&test_data[10..])),
        ];

        let mock_response = create_mock_response(chunks);

        let parser = |event: &str| -> Result<Option<String>, LLMError> {
            if let Some(content) = event.strip_prefix("data: ") {
                let content = content.trim();
                if content.is_empty() {
                    return Ok(None);
                }
                Ok(Some(content.to_string()))
            } else {
                Ok(None)
            }
        };

        let mut stream = create_sse_stream(mock_response, parser);

        let mut results = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result);
        }

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].as_ref().unwrap(), "Positive reactions");
    }

    #[tokio::test]
    async fn test_create_sse_stream_handles_split_sse_events() {
        let event1 = "data: First event\n\n";
        let event2 = "data: Second event\n\n";
        let combined = format!("{}{}", event1, event2);
        let test_data = combined.as_bytes().to_vec();

        let split_point = event1.len() + 5;
        let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![
            Ok(Bytes::from(test_data[..split_point].to_vec())),
            Ok(Bytes::from(test_data[split_point..].to_vec())),
        ];

        let mock_response = create_mock_response(chunks);

        let parser = |event: &str| -> Result<Option<String>, LLMError> {
            if let Some(content) = event.strip_prefix("data: ") {
                let content = content.trim();
                if content.is_empty() {
                    return Ok(None);
                }
                Ok(Some(content.to_string()))
            } else {
                Ok(None)
            }
        };

        let mut stream = create_sse_stream(mock_response, parser);

        let mut results = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result);
        }

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].as_ref().unwrap(), "First event");
        assert_eq!(results[1].as_ref().unwrap(), "Second event");
    }

    #[tokio::test]
    async fn test_create_sse_stream_handles_multibyte_utf8_split() {
        let multibyte_char = "✨";
        let event = format!("data: Star {}\n\n", multibyte_char);
        let test_data = event.as_bytes().to_vec();

        let emoji_start = event.find(multibyte_char).unwrap();
        let split_in_emoji = emoji_start + 1;

        let chunks: Vec<Result<Bytes, reqwest::Error>> = vec![
            Ok(Bytes::from(test_data[..split_in_emoji].to_vec())),
            Ok(Bytes::from(test_data[split_in_emoji..].to_vec())),
        ];

        let mock_response = create_mock_response(chunks);

        let parser = |event: &str| -> Result<Option<String>, LLMError> {
            if let Some(content) = event.strip_prefix("data: ") {
                let content = content.trim();
                if content.is_empty() {
                    return Ok(None);
                }
                Ok(Some(content.to_string()))
            } else {
                Ok(None)
            }
        };

        let mut stream = create_sse_stream(mock_response, parser);

        let mut results = Vec::new();
        while let Some(result) = stream.next().await {
            results.push(result);
        }

        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].as_ref().unwrap(),
            &format!("Star {}", multibyte_char)
        );
    }

    fn create_mock_response(chunks: Vec<Result<Bytes, reqwest::Error>>) -> reqwest::Response {
        use http_body_util::StreamBody;
        use reqwest::Body;

        let frame_stream = futures::stream::iter(
            chunks
                .into_iter()
                .map(|chunk| chunk.map(|bytes| hyper::body::Frame::data(bytes))),
        );

        let body = StreamBody::new(frame_stream);
        let body = Body::wrap(body);

        let http_response = http::Response::builder().status(200).body(body).unwrap();

        http_response.into()
    }
}
