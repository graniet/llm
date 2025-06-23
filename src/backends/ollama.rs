//! Ollama API client implementation for chat and completion functionality.
//!
//! This module provides integration with Ollama's local LLM server through its API.

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, StructuredOutputFormat, Tool},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    FunctionCall, ToolCall,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Client for interacting with Ollama's API.
///
/// Provides methods for chat and completion requests using Ollama's models.
pub struct Ollama {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub stream: Option<bool>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    /// JSON schema for structured output
    pub json_schema: Option<StructuredOutputFormat>,
    /// Available tools for function calling
    pub tools: Option<Vec<Tool>>,
    client: Client,
}

/// Request payload for Ollama's chat API endpoint.
#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: String,
    messages: Vec<OllamaChatMessage<'a>>,
    stream: bool,
    options: Option<OllamaOptions>,
    format: Option<OllamaResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
}

#[derive(Serialize)]
struct OllamaOptions {
    top_p: Option<f32>,
    top_k: Option<u32>,
}

/// Individual message in an Ollama chat conversation.
#[derive(Serialize)]
struct OllamaChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// Response from Ollama's API endpoints.
#[derive(Deserialize, Debug)]
struct OllamaResponse {
    content: Option<String>,
    response: Option<String>,
    message: Option<OllamaChatResponseMessage>,
}

impl std::fmt::Display for OllamaResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let empty = String::new();
        let text = self
            .content
            .as_ref()
            .or(self.response.as_ref())
            .or(self.message.as_ref().map(|m| &m.content))
            .unwrap_or(&empty);

        // Write tool calls if present
        if let Some(message) = &self.message {
            if let Some(tool_calls) = &message.tool_calls {
                for tc in tool_calls {
                    writeln!(
                        f,
                        "{{\"name\": \"{}\", \"arguments\": {}}}",
                        tc.function.name,
                        serde_json::to_string_pretty(&tc.function.arguments).unwrap_or_default()
                    )?;
                }
            }
        }

        write!(f, "{}", text)
    }
}

impl ChatResponse for OllamaResponse {
    fn text(&self) -> Option<String> {
        self.content
            .as_ref()
            .or(self.response.as_ref())
            .or(self.message.as_ref().map(|m| &m.content))
            .map(|s| s.to_string())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.message.as_ref().and_then(|msg| {
            msg.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        id: format!("call_{}", tc.function.name),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: tc.function.name.clone(),
                            arguments: serde_json::to_string(&tc.function.arguments)
                                .unwrap_or_default(),
                        },
                    })
                    .collect()
            })
        })
    }
}

/// Message content within an Ollama chat API response.
#[derive(Deserialize, Debug)]
struct OllamaChatResponseMessage {
    content: String,
    tool_calls: Option<Vec<OllamaToolCall>>,
}

/// Request payload for Ollama's generate API endpoint.
#[derive(Serialize)]
struct OllamaGenerateRequest<'a> {
    model: String,
    prompt: &'a str,
    raw: bool,
    stream: bool,
}

#[derive(Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct OllamaEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize, Debug, Serialize)]
#[serde(untagged)]
enum OllamaResponseType {
    #[serde(rename = "json")]
    Json,
    StructuredOutput(Value),
}

#[derive(Deserialize, Debug, Serialize)]
struct OllamaResponseFormat {
    #[serde(flatten)]
    format: OllamaResponseType,
}

/// Ollama's tool format
#[derive(Serialize, Debug)]
struct OllamaTool {
    #[serde(rename = "type")]
    pub tool_type: String,

    pub function: OllamaFunctionTool,
}

#[derive(Serialize, Debug)]
struct OllamaFunctionTool {
    /// Name of the tool
    name: String,
    /// Description of what the tool does
    description: String,
    /// Parameters for the tool
    parameters: OllamaParameters,
}

impl From<&crate::chat::Tool> for OllamaTool {
    fn from(tool: &crate::chat::Tool) -> Self {
        let properties_value = tool
            .function
            .parameters
            .get("properties")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

        let required_fields = tool
            .function
            .parameters
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();

        OllamaTool {
            tool_type: "function".to_owned(),
            function: OllamaFunctionTool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: OllamaParameters {
                    schema_type: "object".to_string(),
                    properties: properties_value,
                    required: required_fields,
                },
            },
        }
    }
}

/// Ollama's parameters schema
#[derive(Serialize, Debug)]
struct OllamaParameters {
    /// The type of parameters object (usually "object")
    #[serde(rename = "type")]
    schema_type: String,
    /// Map of parameter names to their properties
    properties: Value,
    /// List of required parameter names
    required: Vec<String>,
}

/// Ollama's tool call response
#[derive(Deserialize, Debug)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Deserialize, Debug)]
struct OllamaFunctionCall {
    /// Name of the tool that was called
    name: String,
    /// Arguments provided to the tool
    arguments: Value,
}

impl Ollama {
    /// Creates a new Ollama client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the Ollama server
    /// * `api_key` - Optional API key for authentication
    /// * `model` - Model name to use (defaults to "llama3.1")
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt
    /// * `stream` - Whether to stream responses
    /// * `json_schema` - JSON schema for structured output
    /// * `tools` - Function tools that the model can use
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        json_schema: Option<StructuredOutputFormat>,
        tools: Option<Vec<Tool>>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            base_url: base_url.into(),
            api_key,
            model: model.unwrap_or("llama3.1".to_string()),
            temperature,
            max_tokens,
            timeout_seconds,
            system,
            stream,
            top_p,
            top_k,
            json_schema,
            tools,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

/// Parses a Server-Sent Events (SSE) chunk from Ollama's streaming API.
///
/// Ollama's streaming format sends JSON objects separated by newlines, where each
/// object contains a "message" field with the content delta.
///
/// # Arguments
///
/// * `chunk` - Raw SSE chunk data as a string
///
/// # Returns
///
/// * `Ok(Some(String))` - If content was extracted from the chunk
/// * `Ok(None)` - If the chunk contained no content or was a control message
/// * `Err(LLMError)` - If parsing fails
fn parse_ollama_sse_chunk(chunk: &str) -> Result<Option<String>, LLMError> {
    let mut collected_content = String::new();

    for line in chunk.lines() {
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Ollama sends JSON objects directly, not prefixed with "data: "
        match serde_json::from_str::<OllamaResponse>(line) {
            Ok(response) => {
                // Check if this is the final message (done: true)
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(done) = value.get("done").and_then(|v| v.as_bool()) {
                        if done {
                            // This is the final chunk, return any collected content
                            if collected_content.is_empty() {
                                return Ok(None);
                            } else {
                                return Ok(Some(collected_content));
                            }
                        }
                    }
                }

                // Extract content from the response
                if let Some(content) = response.text() {
                    if !content.is_empty() {
                        collected_content.push_str(&content);
                    }
                }
            }
            Err(_) => {
                // If we can't parse as OllamaResponse, try to parse as a generic JSON
                // to check for the "done" field
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(done) = value.get("done").and_then(|v| v.as_bool()) {
                        if done {
                            if collected_content.is_empty() {
                                return Ok(None);
                            } else {
                                return Ok(Some(collected_content));
                            }
                        }
                    }
                }
                // Skip lines that can't be parsed
                continue;
            }
        }
    }

    if collected_content.is_empty() {
        Ok(None)
    } else {
        Ok(Some(collected_content))
    }
}

#[async_trait]
impl ChatProvider for Ollama {
    /// Sends a chat request to Ollama's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// The model's response text or an error
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }

        let mut chat_messages: Vec<OllamaChatMessage> = messages
            .iter()
            .map(|msg| OllamaChatMessage {
                role: match msg.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &msg.content,
            })
            .collect();

        if let Some(system) = &self.system {
            chat_messages.insert(
                0,
                OllamaChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        // Ollama doesn't require the "name" field in the schema, so we just use the schema itself
        let format = if let Some(schema) = &self.json_schema {
            schema.schema.as_ref().map(|schema| OllamaResponseFormat {
                format: OllamaResponseType::StructuredOutput(schema.clone()),
            })
        } else {
            None
        };

        let req_body = OllamaChatRequest {
            model: self.model.clone(),
            messages: chat_messages,
            stream: self.stream.unwrap_or(false),
            options: Some(OllamaOptions {
                top_p: self.top_p,
                top_k: self.top_k,
            }),
            format,
            tools: None,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Ollama request payload: {}", json);
            }
        }

        let url = format!("{}/api/chat", self.base_url);

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("Ollama HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;
        let json_resp: OllamaResponse = resp.json().await?;
        Ok(Box::new(json_resp))
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }

        let mut chat_messages: Vec<OllamaChatMessage> = messages
            .iter()
            .map(|msg| OllamaChatMessage {
                role: match msg.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &msg.content,
            })
            .collect();

        if let Some(system) = &self.system {
            chat_messages.insert(
                0,
                OllamaChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        // Convert tools to Ollama format if provided
        let ollama_tools = tools.map(|t| t.iter().map(OllamaTool::from).collect());

        // Ollama doesn't require the "name" field in the schema, so we just use the schema itself
        let format = if let Some(schema) = &self.json_schema {
            schema.schema.as_ref().map(|schema| OllamaResponseFormat {
                format: OllamaResponseType::StructuredOutput(schema.clone()),
            })
        } else {
            None
        };

        let req_body = OllamaChatRequest {
            model: self.model.clone(),
            messages: chat_messages,
            stream: self.stream.unwrap_or(false),
            options: Some(OllamaOptions {
                top_p: self.top_p,
                top_k: self.top_k,
            }),
            format,
            tools: ollama_tools,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Ollama request payload (tools): {}", json);
            }
        }

        let url = format!("{}/api/chat", self.base_url);

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("Ollama HTTP status (tools): {}", resp.status());

        let resp = resp.error_for_status()?;
        let json_resp = resp.json::<OllamaResponse>().await?;

        Ok(Box::new(json_resp))
    }

    /// Sends a streaming chat request to Ollama's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// A stream of text tokens or an error
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<String, LLMError>> + Send>>,
        LLMError,
    > {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }

        let mut chat_messages: Vec<OllamaChatMessage> = messages
            .iter()
            .map(|msg| OllamaChatMessage {
                role: match msg.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &msg.content,
            })
            .collect();

        if let Some(system) = &self.system {
            chat_messages.insert(
                0,
                OllamaChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        // Ollama doesn't require the "name" field in the schema, so we just use the schema itself
        let format = if let Some(schema) = &self.json_schema {
            schema.schema.as_ref().map(|schema| OllamaResponseFormat {
                format: OllamaResponseType::StructuredOutput(schema.clone()),
            })
        } else {
            None
        };

        let req_body = OllamaChatRequest {
            model: self.model.clone(),
            messages: chat_messages,
            stream: true, // Enable streaming
            options: Some(OllamaOptions {
                top_p: self.top_p,
                top_k: self.top_k,
            }),
            format,
            tools: None,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&req_body) {
                log::trace!("Ollama streaming request payload: {}", json);
            }
        }

        let url = format!("{}/api/chat", self.base_url);

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("Ollama streaming HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;

        Ok(crate::chat::create_sse_stream(resp, parse_ollama_sse_chunk))
    }
}

#[async_trait]
impl CompletionProvider for Ollama {
    /// Sends a completion request to Ollama's API.
    ///
    /// # Arguments
    ///
    /// * `req` - The completion request containing the prompt
    ///
    /// # Returns
    ///
    /// The completion response containing the generated text or an error
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }
        let url = format!("{}/api/generate", self.base_url);

        let req_body = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt: &req.prompt,
            raw: true,
            stream: false,
        };

        let resp = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await?
            .error_for_status()?;
        let json_resp: OllamaResponse = resp.json().await?;

        if let Some(answer) = json_resp.response.or(json_resp.content) {
            Ok(CompletionResponse { text: answer })
        } else {
            Err(LLMError::ProviderError(
                "No answer returned by Ollama".to_string(),
            ))
        }
    }
}

#[async_trait]
impl EmbeddingProvider for Ollama {
    async fn embed(&self, text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.base_url.is_empty() {
            return Err(LLMError::InvalidRequest("Missing base_url".to_string()));
        }
        let url = format!("{}/api/embed", self.base_url);

        let body = OllamaEmbeddingRequest {
            model: self.model.clone(),
            input: text,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: OllamaEmbeddingResponse = resp.json().await?;
        Ok(json_resp.embeddings)
    }
}

#[async_trait]
impl SpeechToTextProvider for Ollama {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Ollama does not implement speech to text endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for Ollama {}

impl crate::LLMProvider for Ollama {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl TextToSpeechProvider for Ollama {}
