//! Anthropic API client implementation for chat and completion functionality.
//!
//! This module provides integration with Anthropic's Claude models through their API.

use std::collections::HashMap;
use std::fs;

use crate::{
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, ParametersSchema, Tool,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{FunctionCall, ToolCall};

/// Client for interacting with Anthropic's API.
///
/// Provides methods for chat and completion requests using Anthropic's models.
#[derive(Debug)]
pub struct Anthropic {
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_seconds: u64,
    pub system: String,
    pub stream: bool,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tools: Option<Vec<Tool>>,
    pub default_image_type: String,
    pub reasoning: bool,
    pub thinking_budget_tokens: Option<u32>,
    client: Client,
}

/// Anthropic-specific tool format that matches their API structure
#[derive(Serialize, Debug)]
struct AnthropicTool<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a ParametersSchema,
}

/// Configuration for the thinking feature
#[derive(Serialize, Debug)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
    budget_tokens: u32,
}

/// Request payload for Anthropic's messages API endpoint.
#[derive(Serialize, Debug)]
struct AnthropicCompleteRequest<'a> {
    messages: Vec<AnthropicMessage<'a>>,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
}

/// Individual message in an Anthropic chat conversation.
#[derive(Serialize, Debug)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: Vec<MessageContent<'a>>,
}

#[derive(Serialize, Debug)]
struct MessageContent<'a> {
    #[serde(rename = "type")]
    message_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ImageUrlContent<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<ImageSource<'a>>,
}

#[derive(Serialize, Debug)]
struct ImageUrlContent<'a> {
    url: &'a str,
}

#[derive(Serialize, Debug)]
struct ImageSource<'a> {
    #[serde(rename = "type")]
    source_type: &'a str,
    media_type: &'a str,
    data: String,
}

/// Response from Anthropic's messages API endpoint.
#[derive(Deserialize, Debug)]
struct AnthropicCompleteResponse {
    content: Vec<AnthropicContent>,
}

/// Content block within an Anthropic API response.
#[derive(Serialize, Deserialize, Debug)]
struct AnthropicContent {
    text: Option<String>,
    #[serde(rename = "type")]
    content_type: Option<String>,
    thinking: Option<String>,
    name: Option<String>,
    input: Option<HashMap<String, String>>,
    id: Option<String>,
}

impl std::fmt::Display for AnthropicCompleteResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for content in self.content.iter() {
            match content.content_type {
                Some(ref t) if t == "tool_use" => write!(
                    f,
                    "{{\n \"name\": {}, \"input\": {:?}\n}}",
                    content.name.clone().unwrap_or_default(),
                    content.input.clone().unwrap_or_default()
                )?,
                Some(ref t) if t == "thinking" => {
                    write!(f, "{}", content.thinking.clone().unwrap_or_default())?
                }
                _ => write!(
                    f,
                    "{}",
                    self.content
                        .iter()
                        .map(|c| c.text.clone().unwrap_or_default())
                        .collect::<Vec<_>>()
                        .join("\n")
                )?,
            }
        }
        Ok(())
    }
}

impl ChatResponse for AnthropicCompleteResponse {
    fn text(&self) -> Option<String> {
        Some(
            self.content
                .iter()
                .filter_map(|c| {
                    if c.content_type == Some("text".to_string()) || c.content_type.is_none() {
                        c.text.clone()
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    fn thinking(&self) -> Option<String> {
        self.content
            .iter()
            .find(|c| c.content_type == Some("thinking".to_string()))
            .and_then(|c| c.thinking.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        Some(
            self.content
                .iter()
                .filter_map(|c| {
                    if c.content_type == Some("tool_use".to_string()) {
                        Some(ToolCall {
                            id: "".to_string(),
                            call_type: "".to_string(),
                            function: FunctionCall {
                                name: c.name.clone().unwrap_or_default(),
                                arguments: serde_json::to_string(
                                    &c.input.clone().unwrap_or_default(),
                                )
                                .unwrap_or_default(),
                            },
                        })
                    } else {
                        None
                    }
                })
                .collect(),
        )
    }
}

impl Anthropic {
    /// Encodes binary image data to base64
    fn encode_image_base64(data: &[u8]) -> String {
        BASE64.encode(data)
    }

    /// Reads and encodes an image file to base64
    fn read_and_encode_image(path: &str) -> String {
        match fs::read(path) {
            Ok(bytes) => Self::encode_image_base64(&bytes),
            Err(e) => {
                eprintln!("Warning: Failed to read image: {}", e);
                String::new()
            }
        }
    }

    /// Creates a new Anthropic client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Anthropic API key for authentication
    /// * `model` - Model identifier (defaults to "claude-3-sonnet-20240229")
    /// * `max_tokens` - Maximum tokens in response (defaults to 300)
    /// * `temperature` - Sampling temperature (defaults to 0.7)
    /// * `timeout_seconds` - Request timeout in seconds (defaults to 30)
    /// * `system` - System prompt (defaults to "You are a helpful assistant.")
    /// * `stream` - Whether to stream responses (defaults to false)
    /// *
    /// * `thinking_budget_tokens` - Budget tokens for thinking (optional)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Vec<Tool>>,
        reasoning: Option<bool>,
        thinking_budget_tokens: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or_else(|| "claude-3-sonnet-20240229".to_string()),
            max_tokens: max_tokens.unwrap_or(300),
            temperature: temperature.unwrap_or(0.7),
            system: system.unwrap_or_else(|| "You are a helpful assistant.".to_string()),
            timeout_seconds: timeout_seconds.unwrap_or(30),
            stream: stream.unwrap_or(false),
            top_p,
            top_k,
            tools,
            reasoning: reasoning.unwrap_or(false),
            thinking_budget_tokens,
            default_image_type: "image/jpeg".to_string(),
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

#[async_trait]
impl ChatProvider for Anthropic {
    /// Sends a chat request to Anthropic's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools to use in the chat
    ///
    /// # Returns
    ///
    /// The model's response text or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Anthropic API key".to_string()));
        }

        let anthropic_messages: Vec<AnthropicMessage> = messages
            .iter()
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: match m.message_type {
                    MessageType::Text => vec![MessageContent {
                        message_type: Some("text"),
                        text: Some(&m.content),
                        image_url: None,
                        source: None,
                    }],
                    MessageType::Pdf(_) => unimplemented!(),
                    MessageType::Image => {
                        let encoded = Self::read_and_encode_image(&m.content);
                        vec![MessageContent {
                            message_type: Some("image"),
                            text: None,
                            image_url: None,
                            source: Some(ImageSource {
                                source_type: "base64",
                                media_type: &self.default_image_type,
                                data: encoded,
                            }),
                        }]
                    }
                    MessageType::ImageURL => vec![MessageContent {
                        message_type: Some("image_url"),
                        text: None,
                        image_url: Some(ImageUrlContent { url: &m.content }),
                        source: None,
                    }],
                },
            })
            .collect();

        let anthropic_tools = tools.map(|t| {
            t.iter()
                .map(|tool| AnthropicTool {
                    name: &tool.function.name,
                    description: &tool.function.description,
                    input_schema: &tool.function.parameters,
                })
                .collect::<Vec<_>>()
        });

        let tool_choice = if self.tools.is_some() {
            Some(HashMap::from([("type".to_string(), "auto".to_string())]))
        } else {
            None
        };

        let thinking = if self.reasoning {
            Some(ThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: self.thinking_budget_tokens.unwrap_or(16000),
            })
        } else {
            None
        };

        let req_body = AnthropicCompleteRequest {
            messages: anthropic_messages,
            model: &self.model,
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            system: Some(&self.system),
            stream: Some(self.stream),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: anthropic_tools,
            tool_choice,
            thinking,
        };

        let mut request = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&req_body);

        if self.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.timeout_seconds));
        }

        let resp = request.send().await?.error_for_status()?;
        let json_resp: AnthropicCompleteResponse = resp.json().await?;
        Ok(Box::new(json_resp))
    }

    /// Sends a chat request to Anthropic's API.
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
}

#[async_trait]
impl CompletionProvider for Anthropic {
    /// Sends a completion request to Anthropic's API.
    ///
    /// Converts the completion request into a chat message format.
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        unimplemented!()
    }
}

#[async_trait]
impl EmbeddingProvider for Anthropic {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

impl crate::LLMProvider for Anthropic {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}
