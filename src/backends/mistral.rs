//! Mistral API client implementation using the OpenAI-compatible base
//!
//! This module provides integration with Mistral's LLM models through their API.

use crate::builder::LLMBackend;
use crate::models::{ModelListRequest, ModelListResponse, StandardModelListResponse};
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

/// Mistral configuration for the generic provider
pub struct MistralConfig;

impl OpenAIProviderConfig for MistralConfig {
    const PROVIDER_NAME: &'static str = "Mistral";
    const DEFAULT_BASE_URL: &'static str = "https://api.mistral.ai/v1/";
    const DEFAULT_MODEL: &'static str = "mistral-small-latest";
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = true;
}

/// Type alias for Mistral client using the generic OpenAI-compatible provider
pub type Mistral = OpenAICompatibleProvider<MistralConfig>;

impl Mistral {
    /// Creates a new Mistral client with the specified configuration.
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
        <OpenAICompatibleProvider<MistralConfig>>::new(
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
            None, // voice - not supported by Mistral
            extra_body,
            parallel_tool_calls,
            normalize_response,
            embedding_encoding_format,
            embedding_dimensions,
        )
    }
}

// Mistral-specific implementations that don't fit in the generic OpenAI-compatible provider

#[derive(Serialize)]
struct MistralEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct MistralEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct MistralEmbeddingResponse {
    data: Vec<MistralEmbeddingData>,
}

impl LLMProvider for Mistral {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl CompletionProvider for Mistral {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Mistral completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl SpeechToTextProvider for Mistral {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Mistral does not support speech-to-text".into(),
        ))
    }

    async fn transcribe_file(&self, _file_path: &str) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Mistral does not support speech-to-text".into(),
        ))
    }
}

#[cfg(feature = "mistral")]
#[async_trait]
impl EmbeddingProvider for Mistral {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Mistral API key".into()));
        }

        let body = MistralEmbeddingRequest {
            model: self.model.clone(),
            input,
            encoding_format: self.embedding_encoding_format.clone(),
            dimensions: self.embedding_dimensions,
        };

        let url = self
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: MistralEmbeddingResponse = resp.json().await?;
        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

#[async_trait]
impl ModelsProvider for Mistral {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Mistral API key".to_string()));
        }
        let url = format!("{}models", MistralConfig::DEFAULT_BASE_URL);
        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::Mistral,
        };
        Ok(Box::new(result))
    }
}

#[async_trait]
impl TextToSpeechProvider for Mistral {
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::ProviderError(
            "Mistral does not support text-to-speech".into(),
        ))
    }
}
