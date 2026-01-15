// src/backends/bedrock/types.rs
//! Type definitions for AWS Bedrock API requests and responses

use crate::backends::aws::models::BedrockModel;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Request for text completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// The prompt to complete
    pub prompt: String,

    /// Optional model to use (defaults to backend's default model)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<BedrockModel>,

    /// Optional system prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Temperature for sampling (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p for nucleus sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

/// Response from text completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Generated text
    pub text: String,

    /// Model used
    pub model: BedrockModel,

    /// Token usage information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,

    /// Reason for completion finishing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Request for chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Messages in the conversation
    pub messages: Vec<ChatMessage>,

    /// Optional model to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<BedrockModel>,

    /// Optional system prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Available tools for the model to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Top-p for nucleus sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

/// A message in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender
    pub role: String,

    /// Content of the message
    pub content: MessageContent,
}

/// Content of a chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),

    /// Multi-modal content (text, images, tool uses, etc.)
    MultiModal(Vec<ContentPart>),
}

/// Part of multi-modal message content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content
    Text { text: String },

    /// Image content
    Image {
        #[serde(with = "serde_bytes")]
        source: Vec<u8>,
        media_type: String,
    },

    /// Tool use by the model
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },

    /// Tool result from the user
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

/// Response from chat completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The assistant's message
    pub message: ChatMessage,

    /// Model used
    pub model: BedrockModel,

    /// Token usage information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,

    /// Reason for completion finishing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

impl std::fmt::Display for ChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text().unwrap_or_default())
    }
}

use crate::{
    chat::{ChatResponse as ChatResponseTrait, Usage},
    FunctionCall, ToolCall,
};

impl ChatResponseTrait for ChatResponse {
    fn text(&self) -> Option<String> {
        match &self.message.content {
            MessageContent::Text(t) => Some(t.clone()),
            MessageContent::MultiModal(parts) => {
                let texts: Vec<String> = parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect();
                if texts.is_empty() {
                    None
                } else {
                    Some(texts.join(""))
                }
            }
        }
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        match &self.message.content {
            MessageContent::Text(_) => None,
            MessageContent::MultiModal(parts) => {
                let calls: Vec<ToolCall> = parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::ToolUse { id, name, input } => Some(ToolCall {
                            id: id.clone(),
                            function: FunctionCall {
                                name: name.clone(),
                                arguments: input.to_string(),
                            },
                            call_type: "function".to_string(),
                        }),
                        _ => None,
                    })
                    .collect();
                if calls.is_empty() {
                    None
                } else {
                    Some(calls)
                }
            }
        }
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.as_ref().map(|u| Usage {
            prompt_tokens: u.input_tokens as u32,
            completion_tokens: u.output_tokens as u32,
            total_tokens: u.total_tokens as u32,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        })
    }
}

/// Chunk of streaming chat response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStreamChunk {
    /// Text delta
    pub delta: String,

    /// Finish reason if stream is complete
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Name of the tool
    pub name: String,

    /// Description of what the tool does
    pub description: String,

    /// JSON schema for the tool's input parameters
    pub input_schema: Value,
}

/// Request for text embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Text to embed
    pub input: String,

    /// Optional model to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<BedrockModel>,

    /// Number of dimensions (model-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,

    /// Whether to normalize the embedding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalize: Option<bool>,

    /// Input type for Cohere models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
}

/// Response from embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The embedding vector
    pub embedding: Vec<f64>,

    /// Model used
    pub model: BedrockModel,

    /// Number of dimensions
    pub dimensions: usize,
}

/// Token usage information
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UsageInfo {
    /// Number of input tokens
    pub input_tokens: u64,

    /// Number of output tokens
    pub output_tokens: u64,

    /// Total tokens
    pub total_tokens: u64,
}

impl ChatMessage {
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::Text(content.into()),
        }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: MessageContent::Text(content.into()),
        }
    }

    /// Create a new user message with image
    pub fn user_with_image(text: String, image_data: Vec<u8>, media_type: String) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::MultiModal(vec![
                ContentPart::Text { text },
                ContentPart::Image {
                    source: image_data,
                    media_type,
                },
            ]),
        }
    }
}

impl CompletionRequest {
    /// Create a new completion request with a prompt
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            model: None,
            system: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
        }
    }

    /// Set the model
    pub fn with_model(mut self, model: BedrockModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the system prompt
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

impl ChatRequest {
    /// Create a new chat request
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            model: None,
            system: None,
            tools: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
        }
    }

    /// Set the model
    pub fn with_model(mut self, model: BedrockModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the system prompt
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Add tools
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl EmbeddingRequest {
    /// Create a new embedding request
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            model: None,
            dimensions: None,
            normalize: None,
            input_type: None,
        }
    }

    /// Set the model
    pub fn with_model(mut self, model: BedrockModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Set dimensions
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::backends::aws::models::{CrossRegionModel, DirectModel};

    use super::*;

    #[test]
    fn test_completion_request_builder() {
        let request = CompletionRequest::new("Hello, world!")
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
            .with_max_tokens(100)
            .with_temperature(0.7);

        assert_eq!(request.prompt, "Hello, world!");
        assert_eq!(
            request.model,
            Some(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
        );
        assert_eq!(request.max_tokens, Some(100));
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_chat_message_creation() {
        let user_msg = ChatMessage::user("Hello");
        assert_eq!(user_msg.role, "user");

        let assistant_msg = ChatMessage::assistant("Hi there");
        assert_eq!(assistant_msg.role, "assistant");
    }

    #[test]
    fn test_message_with_image() {
        let msg = ChatMessage::user_with_image(
            "What's in this image?".to_string(),
            vec![1, 2, 3, 4],
            "image/png".to_string(),
        );

        assert_eq!(msg.role, "user");
        match msg.content {
            MessageContent::MultiModal(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(parts[0], ContentPart::Text { .. }));
                assert!(matches!(parts[1], ContentPart::Image { .. }));
            }
            _ => panic!("Expected multimodal content"),
        }
    }

    #[test]
    fn test_embedding_request_builder() {
        let request = EmbeddingRequest::new("test text")
            .with_model(BedrockModel::Direct(DirectModel::TitanEmbedV2))
            .with_dimensions(512);

        assert_eq!(request.input, "test text");
        assert_eq!(
            request.model,
            Some(BedrockModel::Direct(DirectModel::TitanEmbedV2))
        );
        assert_eq!(request.dimensions, Some(512));
    }
}
