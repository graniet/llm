//! X.AI API client implementation for chat and completion functionality.
//!
//! This module provides integration with X.AI's models through their API.
//! It implements chat and completion capabilities using the X.AI API endpoints.

use std::sync::Arc;

use crate::ToolCall;
#[cfg(feature = "xai")]
use crate::{
    builder::LLMBackend,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, StructuredOutputFormat, Tool, Usage,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider, StandardModelListResponse},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};
use async_trait::async_trait;
use futures::stream::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Configuration for the XAI client.
/// Configuration for the X.AI client.
#[derive(Debug)]
pub struct XAIConfig {
    /// API key for authentication with X.AI.
    pub api_key: String,
    /// Model identifier.
    pub model: String,
    /// Maximum tokens to generate in responses.
    pub max_tokens: Option<u32>,
    /// Sampling temperature for response randomness.
    pub temperature: Option<f32>,
    /// System prompt to guide model behavior.
    pub system: Option<String>,
    /// Request timeout in seconds.
    pub timeout_seconds: Option<u64>,
    /// Top-p (nucleus) sampling parameter.
    pub top_p: Option<f32>,
    /// Top-k sampling parameter.
    pub top_k: Option<u32>,
    /// Encoding format for embeddings.
    pub embedding_encoding_format: Option<String>,
    /// Dimensions for embeddings.
    pub embedding_dimensions: Option<u32>,
    /// JSON schema for structured output.
    pub json_schema: Option<StructuredOutputFormat>,
    /// Search mode for web search functionality.
    pub xai_search_mode: Option<String>,
    /// Source type for search.
    pub xai_search_source_type: Option<String>,
    /// Websites to exclude from search.
    pub xai_search_excluded_websites: Option<Vec<String>>,
    /// Maximum number of search results.
    pub xai_search_max_results: Option<u32>,
    /// Start date for search results.
    pub xai_search_from_date: Option<String>,
    /// End date for search results.
    pub xai_search_to_date: Option<String>,
}

/// Client for interacting with X.AI's API.
///
/// This struct provides methods for making chat and completion requests to X.AI's language models.
/// It handles authentication, request configuration, and response parsing.
///
/// The client uses `Arc` internally for configuration, making cloning cheap.
#[derive(Debug, Clone)]
pub struct XAI {
    /// Shared configuration wrapped in Arc for cheap cloning.
    pub config: Arc<XAIConfig>,
    /// HTTP client for making requests.
    pub client: Client,
}

/// Search source configuration for search parameters
#[derive(Debug, Clone, serde::Serialize)]
pub struct XaiSearchSource {
    /// Type of source: "web" or "news"
    #[serde(rename = "type")]
    pub source_type: String,
    /// List of websites to exclude from this source
    pub excluded_websites: Option<Vec<String>>,
}

/// Search parameters for LLM providers that support search functionality
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct XaiSearchParameters {
    /// Search mode (e.g., "auto")
    pub mode: Option<String>,
    /// List of search sources with exclusions
    pub sources: Option<Vec<XaiSearchSource>>,
    /// Maximum number of search results to return
    pub max_search_results: Option<u32>,
    /// Start date for search results (format: "YYYY-MM-DD")
    pub from_date: Option<String>,
    /// End date for search results (format: "YYYY-MM-DD")
    pub to_date: Option<String>,
}

/// Individual message in an X.AI chat conversation.
#[derive(Serialize)]
struct XAIChatMessage<'a> {
    /// Role of the message sender (user, assistant, or system)
    role: &'a str,
    /// Content of the message
    content: &'a str,
}

/// Request payload for X.AI's chat API endpoint.
#[derive(Serialize)]
struct XAIChatRequest<'a> {
    /// Model identifier to use
    model: &'a str,
    /// Array of conversation messages
    messages: Vec<XAIChatMessage<'a>>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// Temperature parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Whether to stream the response
    stream: bool,
    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    /// Top-k sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<XAIResponseFormat>,
    /// Search parameters for search functionality
    #[serde(skip_serializing_if = "Option::is_none")]
    search_parameters: Option<&'a XaiSearchParameters>,
}

/// Response from X.AI's chat API endpoint.
#[derive(Deserialize, Debug)]
struct XAIChatResponse {
    /// Array of generated responses
    choices: Vec<XAIChatChoice>,
    /// Usage metadata for the request
    usage: Option<Usage>,
}

impl std::fmt::Display for XAIChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text().unwrap_or_default())
    }
}

impl ChatResponse for XAIChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().map(|c| c.message.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

/// Individual response choice from the chat API.
#[derive(Deserialize, Debug)]
struct XAIChatChoice {
    /// Message content and metadata
    message: XAIChatMsg,
}

/// Message content from a chat response.
#[derive(Deserialize, Debug)]
struct XAIChatMsg {
    /// Generated text content
    content: String,
}

#[derive(Debug, Serialize)]
struct XAIEmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize)]
struct XAIEmbeddingData {
    embedding: Vec<f32>,
}

/// Response from X.AI's streaming chat API endpoint.
#[derive(Deserialize, Debug)]
struct XAIStreamResponse {
    /// Array of generated responses
    choices: Vec<XAIStreamChoice>,
}

/// Individual response choice from the streaming chat API.
#[derive(Deserialize, Debug)]
struct XAIStreamChoice {
    /// Delta content
    delta: XAIStreamDelta,
}

/// Delta content from a streaming chat response.
#[derive(Deserialize, Debug)]
struct XAIStreamDelta {
    /// Generated text content
    content: Option<String>,
}

#[derive(Deserialize)]
struct XAIEmbeddingResponse {
    data: Vec<XAIEmbeddingData>,
}

#[derive(Deserialize, Debug, Serialize)]
enum XAIResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

/// An object specifying the format that the model must output.
/// Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured Outputs which ensures the model will match your supplied JSON schema. Learn more in the [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).
/// Setting to `{ "type": "json_object" }` enables the older JSON mode, which ensures the message the model generates is valid JSON. Using `json_schema` is preferred for models that support it.
/// The structured outputs feature is only supported for the `grok-2-latest` model.
#[derive(Deserialize, Debug, Serialize)]
struct XAIResponseFormat {
    #[serde(rename = "type")]
    response_type: XAIResponseType,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<StructuredOutputFormat>,
}

impl XAI {
    /// Creates a new X.AI client with the specified configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        json_schema: Option<StructuredOutputFormat>,
        xai_search_mode: Option<String>,
        xai_search_source_type: Option<String>,
        xai_search_excluded_websites: Option<Vec<String>>,
        xai_search_max_results: Option<u32>,
        xai_search_from_date: Option<String>,
        xai_search_to_date: Option<String>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self::with_client(
            builder.build().expect("Failed to build reqwest Client"),
            api_key,
            model,
            max_tokens,
            temperature,
            timeout_seconds,
            system,
            top_p,
            top_k,
            embedding_encoding_format,
            embedding_dimensions,
            json_schema,
            xai_search_mode,
            xai_search_source_type,
            xai_search_excluded_websites,
            xai_search_max_results,
            xai_search_from_date,
            xai_search_to_date,
        )
    }

    /// Creates a new X.AI client with a custom HTTP client.
    #[allow(clippy::too_many_arguments)]
    pub fn with_client(
        client: Client,
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        json_schema: Option<StructuredOutputFormat>,
        xai_search_mode: Option<String>,
        xai_search_source_type: Option<String>,
        xai_search_excluded_websites: Option<Vec<String>>,
        xai_search_max_results: Option<u32>,
        xai_search_from_date: Option<String>,
        xai_search_to_date: Option<String>,
    ) -> Self {
        Self {
            config: Arc::new(XAIConfig {
                api_key: api_key.into(),
                model: model.unwrap_or("grok-2-latest".to_string()),
                max_tokens,
                temperature,
                system,
                timeout_seconds,
                top_p,
                top_k,
                embedding_encoding_format,
                embedding_dimensions,
                json_schema,
                xai_search_mode,
                xai_search_source_type,
                xai_search_excluded_websites,
                xai_search_max_results,
                xai_search_from_date,
                xai_search_to_date,
            }),
            client,
        }
    }

    pub fn api_key(&self) -> &str {
        &self.config.api_key
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    pub fn max_tokens(&self) -> Option<u32> {
        self.config.max_tokens
    }

    pub fn temperature(&self) -> Option<f32> {
        self.config.temperature
    }

    pub fn timeout_seconds(&self) -> Option<u64> {
        self.config.timeout_seconds
    }

    pub fn system(&self) -> Option<&str> {
        self.config.system.as_deref()
    }

    pub fn top_p(&self) -> Option<f32> {
        self.config.top_p
    }

    pub fn top_k(&self) -> Option<u32> {
        self.config.top_k
    }

    pub fn embedding_encoding_format(&self) -> Option<&str> {
        self.config.embedding_encoding_format.as_deref()
    }

    pub fn embedding_dimensions(&self) -> Option<u32> {
        self.config.embedding_dimensions
    }

    pub fn json_schema(&self) -> Option<&StructuredOutputFormat> {
        self.config.json_schema.as_ref()
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
}

#[async_trait]
impl ChatProvider for XAI {
    /// Sends a chat request to the X.AI API and returns the response.
    ///
    /// # Arguments
    ///
    /// * `messages` - Array of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// The generated response text, or an error if the request fails.
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.config.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing X.AI API key".to_string()));
        }

        let mut xai_msgs: Vec<XAIChatMessage> = messages
            .iter()
            .map(|m| XAIChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        if let Some(system) = &self.config.system {
            xai_msgs.insert(
                0,
                XAIChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        // OpenAI's structured output has some [odd requirements](https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat&lang=curl#supported-schemas).
        // There's currently no check for these, so we'll leave it up to the user to provide a valid schema.
        // Unknown if XAI requires these too, but since it copies everything else from OpenAI, it's likely.
        let response_format: Option<XAIResponseFormat> =
            self.config.json_schema.as_ref().map(|s| XAIResponseFormat {
                response_type: XAIResponseType::JsonSchema,
                json_schema: Some(s.clone()),
            });

        let search_parameters = XaiSearchParameters {
            mode: self.config.xai_search_mode.clone(),
            sources: Some(vec![XaiSearchSource {
                source_type: self
                    .config
                    .xai_search_source_type
                    .clone()
                    .unwrap_or("web".to_string()),
                excluded_websites: self.config.xai_search_excluded_websites.clone(),
            }]),
            max_search_results: self.config.xai_search_max_results,
            from_date: self.config.xai_search_from_date.clone(),
            to_date: self.config.xai_search_to_date.clone(),
        };

        let body = XAIChatRequest {
            model: &self.config.model,
            messages: xai_msgs,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            stream: false,
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            response_format,
            search_parameters: Some(&search_parameters),
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("XAI request payload: {}", json);
            }
        }

        let mut request = self
            .client
            .post("https://api.x.ai/v1/chat/completions")
            .bearer_auth(&self.config.api_key)
            .json(&body);

        if let Some(timeout) = self.config.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("XAI HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;

        let json_resp: XAIChatResponse = resp.json().await?;
        Ok(Box::new(json_resp))
    }

    /// Sends a chat request to X.AI's API with tools.
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
        _tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        // XAI doesn't support tools yet, fall back to regular chat
        self.chat(messages).await
    }

    /// Sends a streaming chat request to X.AI's API.
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
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        if self.config.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing X.AI API key".to_string()));
        }

        let mut xai_msgs: Vec<XAIChatMessage> = messages
            .iter()
            .map(|m| XAIChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        if let Some(system) = &self.config.system {
            xai_msgs.insert(
                0,
                XAIChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        let body = XAIChatRequest {
            model: &self.config.model,
            messages: xai_msgs,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            stream: true,
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            response_format: None,
            search_parameters: None,
        };

        let mut request = self
            .client
            .post("https://api.x.ai/v1/chat/completions")
            .bearer_auth(&self.config.api_key)
            .json(&body);

        if let Some(timeout) = self.config.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("X.AI API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(crate::chat::create_sse_stream(
            response,
            parse_xai_sse_chunk,
        ))
    }
}

#[async_trait]
impl CompletionProvider for XAI {
    /// Sends a completion request to X.AI's API.
    ///
    /// This functionality is currently not implemented.
    ///
    /// # Arguments
    ///
    /// * `_req` - The completion request parameters
    ///
    /// # Returns
    ///
    /// A placeholder response indicating the functionality is not implemented.
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "X.AI completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for XAI {
    async fn embed(&self, text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.config.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing X.AI API key".into()));
        }

        let emb_format = self
            .config
            .embedding_encoding_format
            .clone()
            .unwrap_or_else(|| "float".to_string());

        let body = XAIEmbeddingRequest {
            model: &self.config.model,
            input: text,
            encoding_format: Some(&emb_format),
            dimensions: self.config.embedding_dimensions,
        };

        let resp = self
            .client
            .post("https://api.x.ai/v1/embeddings")
            .bearer_auth(&self.config.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: XAIEmbeddingResponse = resp.json().await?;

        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

#[async_trait]
impl SpeechToTextProvider for XAI {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "XAI does not implement speech to text endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for XAI {}

#[async_trait]
impl ModelsProvider for XAI {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        if self.config.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing X.AI API key".to_string()));
        }

        let mut request = self
            .client
            .get("https://api.x.ai/v1/models")
            .bearer_auth(&self.config.api_key);

        if let Some(timeout) = self.config.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?.error_for_status()?;
        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::XAI,
        };
        Ok(Box::new(result))
    }
}

impl LLMProvider for XAI {}

/// Parses a Server-Sent Events (SSE) chunk from X.AI's streaming API.
///
/// # Arguments
///
/// * `chunk` - The raw SSE chunk text
///
/// # Returns
///
/// * `Ok(Some(String))` - Content token if found
/// * `Ok(None)` - If chunk should be skipped (e.g., ping, done signal)
/// * `Err(LLMError)` - If parsing fails
fn parse_xai_sse_chunk(chunk: &str) -> Result<Option<String>, LLMError> {
    for line in chunk.lines() {
        let line = line.trim();

        if let Some(data) = line.strip_prefix("data: ") {
            if data == "[DONE]" {
                return Ok(None);
            }

            match serde_json::from_str::<XAIStreamResponse>(data) {
                Ok(response) => {
                    if let Some(choice) = response.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            return Ok(Some(content.clone()));
                        }
                    }
                    return Ok(None);
                }
                Err(_) => continue,
            }
        }
    }

    Ok(None)
}
