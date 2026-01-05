//! HuggingFace Inference Providers API client implementation for chat functionality.
//!
//! https://huggingface.co/docs/inference-providers

use crate::{
    builder::LLMBackend,
    chat::{StructuredOutputFormat, Tool, ToolChoice},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider, StandardModelListResponse},
    providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};
use async_trait::async_trait;

/// HuggingFace configuration for the generic provider
#[derive(Clone)]
pub struct HuggingFaceConfig;

impl OpenAIProviderConfig for HuggingFaceConfig {
    const PROVIDER_NAME: &'static str = "HuggingFace Inference Providers";
    const DEFAULT_BASE_URL: &'static str = "https://router.huggingface.co/v1/";
    const DEFAULT_MODEL: &'static str = "openai/gpt-oss-20b";
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
}

pub type HuggingFace = OpenAICompatibleProvider<HuggingFaceConfig>;

impl HuggingFace {
    /// Creates a new HuggingFace client with the specified configuration.
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
        _embedding_encoding_format: Option<String>,
        _embedding_dimensions: Option<u32>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
        parallel_tool_calls: Option<bool>,
        normalize_response: Option<bool>,
    ) -> Self {
        OpenAICompatibleProvider::<HuggingFaceConfig>::new(
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
            None, // voice
            extra_body,
            parallel_tool_calls,
            normalize_response,
            None, // embedding_encoding_format
            None, // embedding_dimensions
        )
    }

    /// Creates a new HuggingFace client with a pre-configured HTTP client.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config_and_client(
        client: reqwest::Client,
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
        _embedding_encoding_format: Option<String>,
        _embedding_dimensions: Option<u32>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
        parallel_tool_calls: Option<bool>,
        normalize_response: Option<bool>,
    ) -> Self {
        OpenAICompatibleProvider::<HuggingFaceConfig>::with_client(
            client,
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
            None,
            extra_body,
            parallel_tool_calls,
            normalize_response,
            None,
            None,
        )
    }
}

impl LLMProvider for HuggingFace {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl CompletionProvider for HuggingFace {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "HuggingFace completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for HuggingFace {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl SpeechToTextProvider for HuggingFace {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "HuggingFace does not implement speech to text endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for HuggingFace {}

#[async_trait]
impl ModelsProvider for HuggingFace {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError(
                "Missing HuggingFace API key".to_string(),
            ));
        }

        let url = format!("{}/models", HuggingFaceConfig::DEFAULT_BASE_URL);

        let resp = self
            .client
            .get(&url)
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::HuggingFace,
        };
        Ok(Box::new(result))
    }
}
