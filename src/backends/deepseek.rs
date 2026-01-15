//! DeepSeek API client implementation for chat and completion functionality.
//!
//! This module provides integration with DeepSeek's models through their API.

use std::sync::Arc;

use crate::chat::{ChatResponse, Tool};
#[cfg(feature = "deepseek")]
use crate::{
    builder::LLMBackend,
    chat::{ChatMessage, ChatProvider, ChatRole},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider, StandardModelListResponse},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::ToolCall;

/// Configuration for the DeepSeek client.
#[derive(Debug)]
pub struct DeepSeekConfig {
    /// API key for authentication with DeepSeek.
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
}

/// Client for interacting with DeepSeek's API.
///
/// The client uses `Arc` internally for configuration, making cloning cheap.
#[derive(Debug, Clone)]
pub struct DeepSeek {
    /// Shared configuration wrapped in Arc for cheap cloning.
    pub config: Arc<DeepSeekConfig>,
    /// HTTP client for making requests.
    pub client: Client,
}

#[derive(Serialize)]
struct DeepSeekChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct DeepSeekChatRequest<'a> {
    model: &'a str,
    messages: Vec<DeepSeekChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Deserialize, Debug)]
struct DeepSeekChatResponse {
    choices: Vec<DeepSeekChatChoice>,
}

impl std::fmt::Display for DeepSeekChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Deserialize, Debug)]
struct DeepSeekChatChoice {
    message: DeepSeekChatMsg,
}

#[derive(Deserialize, Debug)]
struct DeepSeekChatMsg {
    content: String,
}
impl ChatResponse for DeepSeekChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| {
            if c.message.content.is_empty() {
                None
            } else {
                Some(c.message.content.clone())
            }
        })
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None
    }
}

impl DeepSeek {
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
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
        )
    }

    /// Creates a new DeepSeek client with a custom HTTP client.
    pub fn with_client(
        client: Client,
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
    ) -> Self {
        Self {
            config: Arc::new(DeepSeekConfig {
                api_key: api_key.into(),
                model: model.unwrap_or("deepseek-chat".to_string()),
                max_tokens,
                temperature,
                system,
                timeout_seconds,
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

    pub fn client(&self) -> &Client {
        &self.client
    }
}

#[async_trait]
impl ChatProvider for DeepSeek {
    /// Sends a chat request to DeepSeek's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of chat messages
    ///
    /// # Returns
    ///
    /// The provider's response text or an error
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.config.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing DeepSeek API key".to_string()));
        }

        let mut deepseek_msgs: Vec<DeepSeekChatMessage> = messages
            .iter()
            .map(|m| DeepSeekChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        if let Some(system) = &self.config.system {
            deepseek_msgs.insert(
                0,
                DeepSeekChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        let body = DeepSeekChatRequest {
            model: &self.config.model,
            messages: deepseek_msgs,
            temperature: self.config.temperature,
            stream: false,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("DeepSeek request payload: {}", json);
            }
        }

        let mut request = self
            .client
            .post("https://api.deepseek.com/v1/chat/completions")
            .bearer_auth(&self.config.api_key)
            .json(&body);

        if let Some(timeout) = self.config.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("DeepSeek HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;

        let json_resp: DeepSeekChatResponse = resp.json().await?;

        Ok(Box::new(json_resp))
    }

    /// Sends a chat request to DeepSeek's API with tools.
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
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        todo!()
    }
}

#[async_trait]
impl CompletionProvider for DeepSeek {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "DeepSeek completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for DeepSeek {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl SpeechToTextProvider for DeepSeek {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "DeepSeek does not implement speech to text endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for DeepSeek {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        if self.config.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing DeepSeek API key".to_string()));
        }

        let resp = self
            .client
            .get("https://api.deepseek.com/v1/models")
            .bearer_auth(&self.config.api_key)
            .send()
            .await?
            .error_for_status()?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::DeepSeek,
        };
        Ok(Box::new(result))
    }
}

impl LLMProvider for DeepSeek {}

#[async_trait]
impl TextToSpeechProvider for DeepSeek {
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::ProviderError(
            "Text to speech not supported".to_string(),
        ))
    }
}
