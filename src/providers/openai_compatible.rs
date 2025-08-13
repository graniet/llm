//! OpenAI-compatible API client base implementation
//!
//! This module provides a generic base for OpenAI-compatible APIs that can be reused
//! across multiple providers like OpenAI, Mistral, XAI, Groq, DeepSeek, etc.

use crate::error::LLMError;
use crate::{
    chat::ChatResponse,
    chat::{
        ChatMessage, ChatProvider, ChatRole, MessageType, StreamChoice, StreamDelta,
        StreamResponse, StructuredOutputFormat, Tool, ToolChoice, Usage,
    },
    ToolCall,
};
use async_trait::async_trait;
use either::*;
use futures::{stream::Stream, StreamExt};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Generic OpenAI-compatible provider
///
/// This struct provides a base implementation for any OpenAI-compatible API.
/// Different providers can customize behavior by implementing the `OpenAICompatibleConfig` trait.
pub struct OpenAICompatibleProvider<T: OpenAIProviderConfig> {
    pub api_key: String,
    pub base_url: Url,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub stream: bool,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning_effort: Option<String>,
    pub json_schema: Option<StructuredOutputFormat>,
    pub voice: Option<String>,
    pub parallel_tool_calls: bool,
    pub embedding_encoding_format: Option<String>,
    pub embedding_dimensions: Option<u32>,
    pub client: Client,
    _phantom: PhantomData<T>,
}

/// Configuration trait for OpenAI-compatible providers
///
/// This trait allows different providers to customize behavior while reusing
/// the common OpenAI-compatible implementation.
pub trait OpenAIProviderConfig: Send + Sync {
    /// The name of the provider (e.g., "OpenAI", "Mistral", "XAI")
    const PROVIDER_NAME: &'static str;
    /// Default base URL for the provider
    const DEFAULT_BASE_URL: &'static str;
    /// Default model for the provider
    const DEFAULT_MODEL: &'static str;
    /// Chat completions endpoint path (usually "chat/completions")
    const CHAT_ENDPOINT: &'static str = "chat/completions";
    /// Whether this provider supports reasoning effort
    const SUPPORTS_REASONING_EFFORT: bool = false;
    /// Whether this provider supports structured output
    const SUPPORTS_STRUCTURED_OUTPUT: bool = false;
    /// Whether this provider supports parallel tool calls
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    /// Whether this provider supports stream options (like include_usage)
    const SUPPORTS_STREAM_OPTIONS: bool = false;
    /// Custom headers to add to requests
    fn custom_headers() -> Option<Vec<(String, String)>> {
        None
    }
}

/// Generic OpenAI-compatible chat message
#[derive(Serialize, Debug)]
pub struct OpenAIChatMessage<'a> {
    pub role: &'a str,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "either::serde_untagged_optional"
    )]
    pub content: Option<Either<Vec<OpenAIMessageContent<'a>>, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIFunctionCall<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct OpenAIFunctionPayload {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Debug)]
pub struct OpenAIFunctionCall<'a> {
    pub id: String,
    #[serde(rename = "type")]
    pub content_type: &'a str,
    pub function: OpenAIFunctionPayload,
}

#[derive(Serialize, Debug)]
pub struct OpenAIMessageContent<'a> {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub message_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrlContent>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_call_id")]
    pub tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    pub tool_output: Option<&'a str>,
}

#[derive(Serialize, Debug)]
pub struct ImageUrlContent {
    pub url: String,
}

/// Generic OpenAI-compatible chat request
#[derive(Serialize, Debug)]
pub struct OpenAIChatRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<OpenAIChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<OpenAIStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// Generic OpenAI-compatible chat response
#[derive(Deserialize, Debug)]
pub struct OpenAIChatResponse {
    pub choices: Vec<OpenAIChatChoice>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIChatChoice {
    pub message: OpenAIChatMsg,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIChatMsg {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug, Serialize)]
pub enum OpenAIResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct OpenAIResponseFormat {
    #[serde(rename = "type")]
    pub response_type: OpenAIResponseType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<StructuredOutputFormat>,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct OpenAIStreamOptions {
    pub include_usage: bool,
}

/// Streaming response structures
#[derive(Deserialize, Debug)]
pub struct ChatStreamChunk {
    pub choices: Vec<ChatStreamChoice>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct ChatStreamChoice {
    pub delta: ChatStreamDelta,
}

#[derive(Deserialize, Debug)]
pub struct ChatStreamDelta {
    pub content: Option<String>,
}

impl From<StructuredOutputFormat> for OpenAIResponseFormat {
    fn from(structured_response_format: StructuredOutputFormat) -> Self {
        match structured_response_format.schema {
            None => OpenAIResponseFormat {
                response_type: OpenAIResponseType::JsonSchema,
                json_schema: Some(structured_response_format),
            },
            Some(mut schema) => {
                schema = if schema.get("additionalProperties").is_none() {
                    schema["additionalProperties"] = serde_json::json!(false);
                    schema
                } else {
                    schema
                };
                OpenAIResponseFormat {
                    response_type: OpenAIResponseType::JsonSchema,
                    json_schema: Some(StructuredOutputFormat {
                        name: structured_response_format.name,
                        description: structured_response_format.description,
                        schema: Some(schema),
                        strict: structured_response_format.strict,
                    }),
                }
            }
        }
    }
}

impl ChatResponse for OpenAIChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| c.message.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.clone())
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

impl std::fmt::Display for OpenAIChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (
            &self.choices.first().unwrap().message.content,
            &self.choices.first().unwrap().message.tool_calls,
        ) {
            (Some(content), Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                write!(f, "{content}")
            }
            (Some(content), None) => write!(f, "{content}"),
            (None, Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                Ok(())
            }
            (None, None) => write!(f, ""),
        }
    }
}

impl<T: OpenAIProviderConfig> OpenAICompatibleProvider<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
        voice: Option<String>,
        parallel_tool_calls: Option<bool>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            base_url: Url::parse(&base_url.unwrap_or_else(|| T::DEFAULT_BASE_URL.to_owned()))
                .expect("Failed to parse base URL"),
            model: model.unwrap_or_else(|| T::DEFAULT_MODEL.to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream: stream.unwrap_or(false),
            top_p,
            top_k,
            tools,
            tool_choice,
            reasoning_effort,
            json_schema,
            voice,
            parallel_tool_calls: parallel_tool_calls.unwrap_or(false),
            embedding_encoding_format,
            embedding_dimensions,
            client: builder.build().expect("Failed to build reqwest Client"),
            _phantom: PhantomData,
        }
    }

    pub fn prepare_messages(&self, messages: &[ChatMessage]) -> Vec<OpenAIChatMessage<'_>> {
        let mut openai_msgs: Vec<OpenAIChatMessage> = messages
            .iter()
            .flat_map(|msg| {
                if let MessageType::ToolResult(ref results) = msg.message_type {
                    // Expand ToolResult into multiple messages
                    results
                        .iter()
                        .map(|result| OpenAIChatMessage {
                            role: "tool",
                            tool_call_id: Some(result.id.clone()),
                            tool_calls: None,
                            content: Some(Right(result.function.arguments.clone())),
                        })
                        .collect::<Vec<_>>()
                } else {
                    // Convert single message
                    vec![chat_message_to_openai_message(msg.clone())]
                }
            })
            .collect();
        if let Some(system) = &self.system {
            openai_msgs.insert(
                0,
                OpenAIChatMessage {
                    role: "system",
                    content: Some(Left(vec![OpenAIMessageContent {
                        message_type: Some("text"),
                        text: Some(system.as_str()),
                        image_url: None,
                        tool_call_id: None,
                        tool_output: None,
                    }])),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }
        openai_msgs
    }
}

#[async_trait]
impl<T: OpenAIProviderConfig> ChatProvider for OpenAICompatibleProvider<T> {
    /// Perform a chat request with tool calls
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError(format!(
                "Missing {} API key",
                T::PROVIDER_NAME
            )));
        }
        let openai_msgs = self.prepare_messages(messages);
        let response_format: Option<OpenAIResponseFormat> = if T::SUPPORTS_STRUCTURED_OUTPUT {
            self.json_schema.clone().map(|s| s.into())
        } else {
            None
        };
        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());
        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };
        let reasoning_effort = if T::SUPPORTS_REASONING_EFFORT {
            self.reasoning_effort.clone()
        } else {
            None
        };
        let parallel_tool_calls = if T::SUPPORTS_PARALLEL_TOOL_CALLS {
            Some(self.parallel_tool_calls)
        } else {
            None
        };
        let body = OpenAIChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: request_tool_choice,
            reasoning_effort,
            response_format,
            stream_options: None,
            parallel_tool_calls,
        };
        let url = self
            .base_url
            .join(T::CHAT_ENDPOINT)
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);
        // Add custom headers if provider specifies them
        if let Some(headers) = T::custom_headers() {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }
        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("{} request payload: {}", T::PROVIDER_NAME, json);
            }
        }
        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        log::debug!("{} HTTP status: {}", T::PROVIDER_NAME, response.status());
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("{} API returned error status: {status}", T::PROVIDER_NAME),
                raw_response: error_text,
            });
        }
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode {} API response: {e}", T::PROVIDER_NAME),
                raw_response: resp_text,
            }),
        }
    }

    /// Perform a chat request without tool calls
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Stream chat responses as a stream of strings
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        let struct_stream = self.chat_stream_struct(messages).await?;
        let content_stream = struct_stream.filter_map(|result| async move {
            match result {
                Ok(stream_response) => {
                    if let Some(choice) = stream_response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            if !content.is_empty() {
                                return Some(Ok(content.clone()));
                            }
                        }
                    }
                    None
                }
                Err(e) => Some(Err(e)),
            }
        });
        Ok(Box::pin(content_stream))
    }

    /// Stream chat responses as `ChatMessage` structured objects, including usage information
    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>,
        LLMError,
    > {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError(format!(
                "Missing {} API key",
                T::PROVIDER_NAME
            )));
        }
        let openai_msgs = self.prepare_messages(messages);
        let body = OpenAIChatRequest {
            model: &self.model,
            messages: openai_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: if T::SUPPORTS_REASONING_EFFORT {
                self.reasoning_effort.clone()
            } else {
                None
            },
            response_format: None,
            stream_options: if T::SUPPORTS_STREAM_OPTIONS {
                Some(OpenAIStreamOptions {
                    include_usage: true,
                })
            } else {
                None
            },
            parallel_tool_calls: if T::SUPPORTS_PARALLEL_TOOL_CALLS {
                Some(self.parallel_tool_calls)
            } else {
                None
            },
        };
        let url = self
            .base_url
            .join(T::CHAT_ENDPOINT)
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);
        if let Some(headers) = T::custom_headers() {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }
        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("{} API returned error status: {status}", T::PROVIDER_NAME),
                raw_response: error_text,
            });
        }
        Ok(create_struct_sse_stream(response))
    }
}

/// Create OpenAICompatibleChatMessage` that doesn't borrow from any temporary variables
pub fn chat_message_to_openai_message(chat_msg: ChatMessage) -> OpenAIChatMessage<'static> {
    OpenAIChatMessage {
        role: match chat_msg.role {
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        },
        tool_call_id: None,
        content: match &chat_msg.message_type {
            MessageType::Text => Some(Right(chat_msg.content.clone())),
            MessageType::Image(_) => unreachable!(),
            MessageType::Pdf(_) => unimplemented!(),
            MessageType::ImageURL(url) => Some(Left(vec![OpenAIMessageContent {
                message_type: Some("image_url"),
                text: None,
                image_url: Some(ImageUrlContent { url: url.clone() }),
                tool_output: None,
                tool_call_id: None,
            }])),
            MessageType::ToolUse(_) => None,
            MessageType::ToolResult(_) => None,
        },
        tool_calls: match &chat_msg.message_type {
            MessageType::ToolUse(calls) => {
                let owned_calls: Vec<OpenAIFunctionCall> = calls
                    .iter()
                    .map(|c| OpenAIFunctionCall {
                        id: c.id.clone(),
                        content_type: "function",
                        function: OpenAIFunctionPayload {
                            name: c.function.name.clone(),
                            arguments: c.function.arguments.clone(),
                        },
                    })
                    .collect();
                Some(owned_calls)
            }
            _ => None,
        },
    }
}

/// Creates a structured SSE stream that returns `StreamResponse` objects
pub fn create_struct_sse_stream(
    response: reqwest::Response,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>> {
    let stream = response
        .bytes_stream()
        .map(move |chunk| match chunk {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                parse_sse_chunk(&text)
            }
            Err(e) => Err(LLMError::HttpError(e.to_string())),
        })
        .filter_map(|result| async move {
            match result {
                Ok(Some(response)) => Some(Ok(response)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        });
    Box::pin(stream)
}

/// Parse SSE chunk and convert to `StreamResponse` format
pub fn parse_sse_chunk(chunk: &str) -> Result<Option<StreamResponse>, LLMError> {
    let mut collected_content = String::new();
    for line in chunk.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                if collected_content.is_empty() {
                    return Ok(None);
                } else {
                    return Ok(Some(StreamResponse {
                        choices: vec![StreamChoice {
                            delta: StreamDelta {
                                content: Some(collected_content),
                            },
                        }],
                        usage: None,
                    }));
                }
            }
            match serde_json::from_str::<ChatStreamChunk>(data) {
                Ok(response) => {
                    if let Some(usage) = response.usage {
                        return Ok(Some(StreamResponse {
                            choices: vec![StreamChoice {
                                delta: StreamDelta { content: None },
                            }],
                            usage: Some(usage),
                        }));
                    }
                    if let Some(choice) = response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            collected_content.push_str(content);
                        }
                    }
                }
                Err(_) => continue,
            }
        }
    }
    if collected_content.is_empty() {
        Ok(None)
    } else {
        Ok(Some(StreamResponse {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: Some(collected_content),
                },
            }],
            usage: None,
        }))
    }
}
