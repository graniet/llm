//! Google Gemini API client implementation for chat and completion functionality.
//!
//! This module provides integration with Google's Gemini models through their API.
//! It implements chat, completion and embedding capabilities via the Gemini API.
//!
//! # Features
//! - Chat conversations with system prompts and message history
//! - Text completion requests
//! - Configuration options for temperature, tokens, top_p, top_k etc.
//! - Streaming support
//!
//! # Example
//! ```no_run
//! use llm::backends::google::Google;
//! use llm::chat::{ChatMessage, ChatRole, ChatProvider, MessageType};
//!
//! #[tokio::main]
//! async fn main() {
//! let client = Google::new(
//!     "your-api-key",
//!     None, // Use default model
//!     Some(1000), // Max tokens
//!     Some(0.7), // Temperature
//!     None, // Default timeout
//!     None, // No system prompt
//!     None, // No streaming
//!     None, // Default top_p
//!     None, // Default top_k
//! );
//!
//! let messages = vec![
//!     ChatMessage {
//!         role: ChatRole::User,
//!         message_type: MessageType::Text,
//!         content: "Hello!".into()
//!     }
//! ];
//!
//! let response = client.chat(&messages).await.unwrap();
//! println!("{}", response);
//! }
//! ```

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, Tool},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    LLMProvider,
};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::ToolCall;

/// Client for interacting with Google's Gemini API.
///
/// This struct holds the configuration and state needed to make requests to the Gemini API.
/// It implements the [`ChatProvider`], [`CompletionProvider`], and [`EmbeddingProvider`] traits.
pub struct Google {
    /// API key for authentication with Google's API
    pub api_key: String,
    /// Model identifier (e.g. "gemini-1.5-flash")
    pub model: String,
    /// Maximum number of tokens to generate in responses
    pub max_tokens: Option<u32>,
    /// Sampling temperature between 0.0 and 1.0
    pub temperature: Option<f32>,
    /// Optional system prompt to set context
    pub system: Option<String>,
    /// Request timeout in seconds
    pub timeout_seconds: Option<u64>,
    /// Whether to stream responses
    pub stream: Option<bool>,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub top_k: Option<u32>,
    /// HTTP client for making API requests
    client: Client,
}

/// Request body for chat completions
#[derive(Serialize)]
struct GoogleChatRequest<'a> {
    /// List of conversation messages
    contents: Vec<GoogleChatContent<'a>>,
    /// Optional generation parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GoogleGenerationConfig>,
}

/// Individual message in a chat conversation
#[derive(Serialize)]
struct GoogleChatContent<'a> {
    /// Role of the message sender ("user", "model", or "system")
    role: &'a str,
    /// Content parts of the message
    parts: Vec<GoogleContentPart<'a>>,
}

/// Text content within a chat message
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum GoogleContentPart<'a> {
    /// The actual text content
    Text(&'a str),
    InlineData(GoogleInlineData),
}

#[derive(Serialize)]
struct GoogleInlineData {
    mime_type: String,
    data: String,
}

/// Configuration parameters for text generation
#[derive(Serialize)]
struct GoogleGenerationConfig {
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    /// Sampling temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    top_p: Option<f32>,
    /// Top-k sampling parameter
    #[serde(skip_serializing_if = "Option::is_none", rename = "topK")]
    top_k: Option<u32>,
}

/// Response from the chat completion API
#[derive(Deserialize, Debug)]
struct GoogleChatResponse {
    /// Generated completion candidates
    candidates: Vec<GoogleCandidate>,
}

impl std::fmt::Display for GoogleChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Individual completion candidate
#[derive(Deserialize, Debug)]
struct GoogleCandidate {
    /// Content of the candidate response
    content: GoogleResponseContent,
}

/// Content block within a response
#[derive(Deserialize, Debug)]
struct GoogleResponseContent {
    /// Parts making up the content
    parts: Vec<GoogleResponsePart>,
}

impl ChatResponse for GoogleChatResponse {
    fn text(&self) -> Option<String> {
        self.candidates
            .first()
            .map(|c| c.content.parts.iter().map(|p| p.text.clone()).collect())
    }
    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        todo!()
    }
}

/// Individual part of response content
#[derive(Deserialize, Debug)]
struct GoogleResponsePart {
    /// Text content of this part
    text: String,
}

/// Request body for embedding content
#[derive(Serialize)]
struct GoogleEmbeddingRequest<'a> {
    model: &'a str,
    content: GoogleEmbeddingContent<'a>,
}

#[derive(Serialize)]
struct GoogleEmbeddingContent<'a> {
    parts: Vec<GoogleContentPart<'a>>,
}

/// Response from the embedding API
#[derive(Deserialize)]
struct GoogleEmbeddingResponse {
    embedding: GoogleEmbedding,
}

#[derive(Deserialize)]
struct GoogleEmbedding {
    values: Vec<f32>,
}

impl Google {
    /// Creates a new Google Gemini client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Google API key for authentication
    /// * `model` - Model identifier (defaults to "gemini-1.5-flash")
    /// * `max_tokens` - Maximum tokens in response
    /// * `temperature` - Sampling temperature between 0.0 and 1.0
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt to set context
    /// * `stream` - Whether to stream responses
    /// * `top_p` - Top-p sampling parameter
    /// * `top_k` - Top-k sampling parameter
    ///
    /// # Returns
    ///
    /// A new `Google` client instance
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
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or_else(|| "gemini-1.5-flash".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

#[async_trait]
impl ChatProvider for Google {
    /// Sends a chat request to Google's Gemini API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// The model's response text or an error
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Google API key".to_string()));
        }

        let mut chat_contents = Vec::with_capacity(messages.len());

        // Add system message if present
        if let Some(system) = &self.system {
            chat_contents.push(GoogleChatContent {
                role: "user",
                parts: vec![GoogleContentPart::Text(system)],
            });
        }

        // Add conversation messages in pairs to maintain context
        for msg in messages {
            chat_contents.push(GoogleChatContent {
                role: match msg.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "model",
                },
                parts: match &msg.message_type {
                    MessageType::Text => vec![GoogleContentPart::Text(&msg.content)],
                    MessageType::Image((image_mime, raw_bytes)) => {
                        vec![GoogleContentPart::InlineData(GoogleInlineData {
                            mime_type: image_mime.mime_type().to_string(),
                            data: BASE64.encode(raw_bytes),
                        })]
                    }
                    MessageType::ImageURL(_) => unimplemented!(),
                    MessageType::Pdf(raw_bytes) => {
                        vec![GoogleContentPart::InlineData(GoogleInlineData {
                            mime_type: "application/pdf".to_string(),
                            data: BASE64.encode(raw_bytes),
                        })]
                    }
                },
            });
        }

        // Remove generation_config if empty to avoid validation errors
        let generation_config = if self.max_tokens.is_none()
            && self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
        {
            None
        } else {
            Some(GoogleGenerationConfig {
                max_output_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
            })
        };

        let req_body = GoogleChatRequest {
            contents: chat_contents,
            generation_config,
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            model = self.model,
            key = self.api_key
        );

        let mut request = self.client.post(&url).json(&req_body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?.error_for_status()?;

        let json_resp: GoogleChatResponse = resp.json().await?;
        Ok(Box::new(json_resp))
    }

    /// Sends a chat request to Google's Gemini API with tools.
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
impl CompletionProvider for Google {
    /// Performs a completion request using the chat endpoint.
    ///
    /// # Arguments
    ///
    /// * `req` - Completion request parameters
    ///
    /// # Returns
    ///
    /// The completion response or an error
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let chat_message = ChatMessage::user().content(req.prompt.clone()).build();
        if let Some(text) = self.chat(&[chat_message]).await?.text() {
            Ok(CompletionResponse { text })
        } else {
            Err(LLMError::ProviderError(
                "No answer returned by Google".to_string(),
            ))
        }
    }
}

#[async_trait]
impl EmbeddingProvider for Google {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Google API key".to_string()));
        }

        let mut embeddings = Vec::new();

        // Process each text separately as Gemini API accepts one text at a time
        for text in texts {
            let req_body = GoogleEmbeddingRequest {
                model: "models/text-embedding-004",
                content: GoogleEmbeddingContent {
                    parts: vec![GoogleContentPart::Text(&text)],
                },
            };

            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={}",
                self.api_key
            );

            let resp = self
                .client
                .post(&url)
                .json(&req_body)
                .send()
                .await?
                .error_for_status()?;

            let embedding_resp: GoogleEmbeddingResponse = resp.json().await?;
            embeddings.push(embedding_resp.embedding.values);
        }

        Ok(embeddings)
    }
}

impl LLMProvider for Google {}
