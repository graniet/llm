//! Cohere API client implementation using the OpenAI-compatible base
//!
//! This module provides integration with Cohere's LLM models through their API.

use crate::providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig};
use crate::{
    chat::{StructuredOutputFormat, Tool, ToolChoice},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Cohere configuration for the generic provider
pub struct CohereConfig;

impl OpenAIProviderConfig for CohereConfig {
    const PROVIDER_NAME: &'static str = "Cohere";
    const DEFAULT_BASE_URL: &'static str = "https://api.cohere.ai/compatibility/v1/";
    // NOTE: upgrading to v2 (not OpenAI-compatible) is required to get usage in streaming responses
    // const DEFAULT_BASE_URL: &'static str = "https://api.cohere.com/v2/chat/";
    const DEFAULT_MODEL: &'static str = "command-r7b-12-2024";
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
}

/// Type alias for Cohere client using the generic OpenAI-compatible provider
pub type Cohere = OpenAICompatibleProvider<CohereConfig>;

impl Cohere {
    /// Creates a new Cohere client with the specified configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        extra_body: Option<serde_json::Value>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
        parallel_tool_calls: Option<bool>,
        normalize_response: Option<bool>,
    ) -> Self {
        <OpenAICompatibleProvider<CohereConfig>>::new(
            api_key,
            base_url,
            model,
            max_tokens,
            temperature,
            timeout_seconds,
            system,
            top_p,
            top_k,
            tools,
            tool_choice,
            reasoning_effort,
            json_schema,
            None, // voice - not supported by Cohere
            extra_body,
            parallel_tool_calls,
            normalize_response,
            embedding_encoding_format,
            embedding_dimensions,
        )
    }
}

// Cohere-specific implementations that don't fit in the generic OpenAI-compatible provider

#[derive(Serialize)]
struct CohereEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct CohereEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct CohereEmbeddingResponse {
    data: Vec<CohereEmbeddingData>,
}

impl LLMProvider for Cohere {
    fn tools(&self) -> Option<&[Tool]> {
        self.config.tools.as_deref()
    }
}

#[async_trait]
impl CompletionProvider for Cohere {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Cohere completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl SpeechToTextProvider for Cohere {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Cohere does not support speech-to-text".into(),
        ))
    }

    async fn transcribe_file(&self, _file_path: &str) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Cohere does not support speech-to-text".into(),
        ))
    }
}

#[cfg(feature = "cohere")]
#[async_trait]
impl EmbeddingProvider for Cohere {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.config.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Cohere API key".into()));
        }

        let body = CohereEmbeddingRequest {
            model: self.config.model.to_owned(),
            input,
            encoding_format: self
                .config
                .embedding_encoding_format
                .as_deref()
                .map(|s| s.to_owned()),
            dimensions: self.config.embedding_dimensions,
        };

        let url = self
            .config
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: CohereEmbeddingResponse = resp.json().await?;
        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

#[async_trait]
impl ModelsProvider for Cohere {
    async fn list_models(
        &self,
        _request: Option<&crate::models::ModelListRequest>,
    ) -> Result<Box<dyn crate::models::ModelListResponse>, LLMError> {
        Err(LLMError::ProviderError(
            "Cohere does not provide a models listing endpoint".into(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for Cohere {
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::ProviderError(
            "Cohere does not support text-to-speech".into(),
        ))
    }
}
