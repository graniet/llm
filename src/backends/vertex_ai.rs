//! Vertex AI Anthropic client (Claude models via Google Cloud).
//!
//! Pass `base_url` as:
//!   https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}
//! Pass `api_key` as a GCP Bearer token (workload identity or ADC).

use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    builder::SystemPrompt,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StreamChunk, Tool,
        ToolChoice, Usage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    FunctionCall, ToolCall,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug)]
pub struct VertexAIConfig {
    /// GCP Bearer token (from workload identity / ADC).
    pub token: String,
    /// https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_seconds: u64,
    pub system: SystemPrompt,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning: bool,
    pub thinking_budget_tokens: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct VertexAI {
    pub config: Arc<VertexAIConfig>,
    pub client: Client,
}

// ── Vertex AI request/response types (Anthropic Messages API format) ──────────

#[derive(Serialize, Debug)]
struct VertexAIRequest<'a> {
    anthropic_version: &'a str,
    messages: Vec<VAMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<VASystemPrompt<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<VATool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<VAThinking>,
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
enum VASystemPrompt<'a> {
    String(&'a str),
    Messages(&'a [crate::builder::SystemContent]),
}

#[derive(Serialize, Debug)]
struct VATool<'a> {
    name: &'a str,
    description: &'a str,
    #[serde(rename = "input_schema")]
    schema: &'a Value,
}

#[derive(Serialize, Debug)]
struct VAThinking {
    #[serde(rename = "type")]
    thinking_type: String,
    budget_tokens: u32,
}

#[derive(Serialize, Debug)]
struct VAMessage<'a> {
    role: &'a str,
    content: Vec<VAContent<'a>>,
}

#[derive(Serialize, Debug)]
struct VAContent<'a> {
    #[serde(rename = "type")]
    content_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<VAImageSource<'a>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "id")]
    tool_use_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "name")]
    tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "input")]
    tool_input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_use_id")]
    tool_result_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    tool_output: Option<String>,
}

#[derive(Serialize, Debug)]
struct VAImageSource<'a> {
    #[serde(rename = "type")]
    source_type: &'a str,
    media_type: &'a str,
    data: String,
}

#[derive(Deserialize, Debug)]
struct VAResponse {
    content: Vec<VAResponseContent>,
    usage: Option<VAUsage>,
}

#[derive(Deserialize, Debug)]
struct VAUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[allow(dead_code)]
    cache_creation_input_tokens: Option<u32>,
    #[allow(dead_code)]
    cache_read_input_tokens: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
struct VAResponseContent {
    text: Option<String>,
    #[serde(rename = "type")]
    content_type: Option<String>,
    thinking: Option<String>,
    name: Option<String>,
    input: Option<Value>,
    id: Option<String>,
}

impl ChatResponse for VAResponse {
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
        let calls: Vec<ToolCall> = self
            .content
            .iter()
            .filter_map(|c| {
                if c.content_type == Some("tool_use".to_string()) {
                    Some(ToolCall {
                        id: c.id.clone().unwrap_or_default(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: c.name.clone().unwrap_or_default(),
                            arguments: serde_json::to_string(
                                &c.input.clone().unwrap_or(Value::Null),
                            )
                            .unwrap_or_default(),
                        },
                    })
                } else {
                    None
                }
            })
            .collect();
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.as_ref().map(|u| Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        })
    }
}

impl std::fmt::Display for VAResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text().unwrap_or_default())
    }
}

// ── VertexAI impl ─────────────────────────────────────────────────────────────

impl VertexAI {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        token: impl Into<String>,
        base_url: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<SystemPrompt>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        reasoning: Option<bool>,
        thinking_budget_tokens: Option<u32>,
    ) -> Self {
        let timeout = timeout_seconds.unwrap_or(60);
        let mut builder = Client::builder();
        if timeout > 0 {
            builder = builder.timeout(std::time::Duration::from_secs(timeout));
        }
        Self {
            config: Arc::new(VertexAIConfig {
                token: token.into(),
                base_url: base_url.into(),
                model: model.unwrap_or_else(|| "claude-sonnet-4-5".to_string()),
                max_tokens: max_tokens.unwrap_or(4096),
                temperature: temperature.unwrap_or(0.7),
                timeout_seconds: timeout,
                system: system.unwrap_or_else(|| {
                    SystemPrompt::String("You are a helpful assistant.".to_string())
                }),
                top_p,
                top_k,
                tools,
                tool_choice,
                reasoning: reasoning.unwrap_or(false),
                thinking_budget_tokens,
            }),
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }

    fn raw_predict_url(&self) -> String {
        format!(
            "{}/publishers/anthropic/models/{}:rawPredict",
            self.config.base_url, self.config.model
        )
    }

    fn stream_raw_predict_url(&self) -> String {
        format!(
            "{}/publishers/anthropic/models/{}:streamRawPredict",
            self.config.base_url, self.config.model
        )
    }

    fn convert_messages<'a>(messages: &'a [ChatMessage]) -> Vec<VAMessage<'a>> {
        messages
            .iter()
            .map(|m| VAMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: match &m.message_type {
                    MessageType::Text => vec![VAContent {
                        content_type: Some("text"),
                        text: Some(&m.content),
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::Image((image_mime, raw_bytes)) => vec![VAContent {
                        content_type: Some("image"),
                        text: None,
                        source: Some(VAImageSource {
                            source_type: "base64",
                            media_type: image_mime.mime_type(),
                            data: BASE64.encode(raw_bytes),
                        }),
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                    MessageType::ToolUse(calls) => calls
                        .iter()
                        .map(|c| VAContent {
                            content_type: Some("tool_use"),
                            text: None,
                            source: None,
                            tool_use_id: Some(c.id.clone()),
                            tool_input: Some(
                                serde_json::from_str(&c.function.arguments)
                                    .unwrap_or(c.function.arguments.clone().into()),
                            ),
                            tool_name: Some(c.function.name.clone()),
                            tool_result_id: None,
                            tool_output: None,
                        })
                        .collect(),
                    MessageType::ToolResult(responses) => responses
                        .iter()
                        .map(|r| VAContent {
                            content_type: Some("tool_result"),
                            text: None,
                            source: None,
                            tool_use_id: None,
                            tool_input: None,
                            tool_name: None,
                            tool_result_id: Some(r.id.clone()),
                            tool_output: Some(r.function.arguments.clone()),
                        })
                        .collect(),
                    _ => vec![VAContent {
                        content_type: Some("text"),
                        text: Some(&m.content),
                        source: None,
                        tool_use_id: None,
                        tool_input: None,
                        tool_name: None,
                        tool_result_id: None,
                        tool_output: None,
                    }],
                },
            })
            .collect()
    }

    fn prepare_tools<'a>(
        tools: Option<&'a [Tool]>,
        instance_tools: Option<&'a [Tool]>,
        tool_choice: &Option<ToolChoice>,
    ) -> (Option<Vec<VATool<'a>>>, Option<HashMap<String, String>>) {
        let maybe_tools: Option<&[Tool]> = tools.or(instance_tools);
        let va_tools = maybe_tools.map(|slice| {
            slice
                .iter()
                .map(|t| VATool {
                    name: &t.function.name,
                    description: &t.function.description,
                    schema: &t.function.parameters,
                })
                .collect()
        });
        let choice_map = match tool_choice {
            Some(ToolChoice::Auto) => {
                Some(HashMap::from([("type".to_string(), "auto".to_string())]))
            }
            Some(ToolChoice::Any) => Some(HashMap::from([("type".to_string(), "any".to_string())])),
            Some(ToolChoice::Tool(name)) => Some(HashMap::from([
                ("type".to_string(), "tool".to_string()),
                ("name".to_string(), name.clone()),
            ])),
            Some(ToolChoice::None) => {
                Some(HashMap::from([("type".to_string(), "none".to_string())]))
            }
            None => None,
        };
        let final_choice = if va_tools.is_some() { choice_map } else { None };
        (va_tools, final_choice)
    }

    fn system_prompt_value<'a>(&'a self) -> Option<VASystemPrompt<'a>> {
        match &self.config.system {
            SystemPrompt::String(s) if !s.is_empty() => Some(VASystemPrompt::String(s)),
            SystemPrompt::Messages(msgs) if !msgs.is_empty() => {
                Some(VASystemPrompt::Messages(msgs))
            }
            _ => None,
        }
    }
}

const AUDIO_UNSUPPORTED: &str = "Audio messages are not supported by Vertex AI";

#[async_trait]
impl ChatProvider for VertexAI {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        crate::chat::ensure_no_audio(messages, AUDIO_UNSUPPORTED)?;

        let va_messages = Self::convert_messages(messages);
        let (va_tools, tool_choice_map) = Self::prepare_tools(
            tools,
            self.config.tools.as_deref(),
            &self.config.tool_choice,
        );

        let thinking = if self.config.reasoning {
            Some(VAThinking {
                thinking_type: "enabled".to_string(),
                budget_tokens: self.config.thinking_budget_tokens.unwrap_or(16000),
            })
        } else {
            None
        };

        let req_body = VertexAIRequest {
            anthropic_version: "vertex-2023-10-16",
            messages: va_messages,
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
            system: self.system_prompt_value(),
            stream: Some(false),
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            tools: va_tools,
            tool_choice: tool_choice_map,
            thinking,
        };

        let mut request = self
            .client
            .post(self.raw_predict_url())
            .header("Authorization", format!("Bearer {}", self.config.token))
            .header("Content-Type", "application/json")
            .json(&req_body);

        if self.config.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.config.timeout_seconds));
        }

        log::debug!("VertexAI request: POST {}", self.raw_predict_url());
        let resp = request.send().await?;
        log::debug!("VertexAI HTTP status: {}", resp.status());

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LLMError::ResponseFormatError {
                message: format!("Vertex AI returned error status: {status}"),
                raw_response: body,
            });
        }

        let body = resp.text().await?;
        let json_resp: VAResponse = serde_json::from_str(&body)
            .map_err(|e| LLMError::HttpError(format!("Failed to parse Vertex AI response: {e}")))?;
        Ok(Box::new(json_resp))
    }

    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        crate::chat::ensure_no_audio(messages, AUDIO_UNSUPPORTED)?;

        let va_messages = Self::convert_messages(messages);

        let req_body = VertexAIRequest {
            anthropic_version: "vertex-2023-10-16",
            messages: va_messages,
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
            system: self.system_prompt_value(),
            stream: Some(true),
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            tools: None,
            tool_choice: None,
            thinking: None,
        };

        let mut request = self
            .client
            .post(self.stream_raw_predict_url())
            .header("Authorization", format!("Bearer {}", self.config.token))
            .header("Content-Type", "application/json")
            .json(&req_body);

        if self.config.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.config.timeout_seconds));
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Vertex AI streaming returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(crate::chat::create_sse_stream(
            response,
            crate::backends::anthropic::parse_anthropic_sse_chunk,
        ))
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError>
    {
        let va_messages = Self::convert_messages(messages);
        let (va_tools, tool_choice_map) = Self::prepare_tools(
            tools,
            self.config.tools.as_deref(),
            &self.config.tool_choice,
        );

        let req_body = VertexAIRequest {
            anthropic_version: "vertex-2023-10-16",
            messages: va_messages,
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
            system: self.system_prompt_value(),
            stream: Some(true),
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            tools: va_tools,
            tool_choice: tool_choice_map,
            thinking: None,
        };

        let mut request = self
            .client
            .post(self.stream_raw_predict_url())
            .header("Authorization", format!("Bearer {}", self.config.token))
            .header("Content-Type", "application/json")
            .json(&req_body);

        if self.config.timeout_seconds > 0 {
            request = request.timeout(std::time::Duration::from_secs(self.config.timeout_seconds));
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Vertex AI streaming returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(crate::backends::anthropic::create_anthropic_tool_stream(
            response,
        ))
    }
}

#[async_trait]
impl CompletionProvider for VertexAI {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Err(LLMError::ProviderError(
            "Completion not supported by Vertex AI backend".to_string(),
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for VertexAI {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported by Vertex AI Anthropic backend".to_string(),
        ))
    }
}

#[async_trait]
impl SpeechToTextProvider for VertexAI {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Speech to text not supported by Vertex AI".to_string(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for VertexAI {}

#[async_trait]
impl ModelsProvider for VertexAI {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        Err(LLMError::ProviderError(
            "Model listing not supported by Vertex AI backend".to_string(),
        ))
    }
}

impl crate::LLMProvider for VertexAI {
    fn tools(&self) -> Option<&[Tool]> {
        self.config.tools.as_deref()
    }
}
