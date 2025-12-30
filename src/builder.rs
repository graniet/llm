//! Builder module for configuring and instantiating LLM providers.
//!
//! This module provides a flexible builder pattern for creating and configuring
//! LLM (Large Language Model) provider instances with various settings and options.

use crate::{
    chat::{
        FunctionTool, ParameterProperty, ParametersSchema, ReasoningEffort, StructuredOutputFormat,
        Tool, ToolChoice,
    },
    error::LLMError,
    memory::{ChatWithMemory, MemoryProvider, SlidingWindowMemory, TrimStrategy},
    LLMProvider,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A function type for validating LLM provider outputs.
/// Takes a response string and returns Ok(()) if valid, or Err with an error message if invalid.
pub type ValidatorFn = dyn Fn(&str) -> Result<(), String> + Send + Sync + 'static;

/// Supported LLM backend providers.
#[derive(Debug, Clone, PartialEq)]
pub enum LLMBackend {
    /// OpenAI API provider (GPT-3, GPT-4, etc.)
    OpenAI,
    /// Anthropic API provider (Claude models)
    Anthropic,
    /// Ollama local LLM provider for self-hosted models
    Ollama,
    /// DeepSeek API provider for their LLM models
    DeepSeek,
    /// X.AI (formerly Twitter) API provider
    XAI,
    /// Phind API provider for code-specialized models
    Phind,
    /// Google Gemini API provider
    Google,
    /// Groq API provider
    Groq,
    /// Azure OpenAI API provider
    AzureOpenAI,
    /// ElevenLabs API provider
    ElevenLabs,
    /// Cohere API provider
    Cohere,
    /// Mistral API provider
    Mistral,
    /// OpenRouter API provider
    OpenRouter,
    /// HuggingFace Inference Providers API
    HuggingFace,
    /// AWS Bedrock API provider
    AwsBedrock
}

/// Implements string parsing for LLMBackend enum.
///
/// Converts a string representation of a backend provider name into the corresponding
/// LLMBackend variant. The parsing is case-insensitive.
///
/// # Arguments
///
/// * `s` - The string to parse
///
/// # Returns
///
/// * `Ok(LLMBackend)` - The corresponding backend variant if valid
/// * `Err(LLMError)` - An error if the string doesn't match any known backend
///
/// # Examples
///
/// ```
/// use std::str::FromStr;
/// use llm::builder::LLMBackend;
///
/// let backend = LLMBackend::from_str("openai").unwrap();
/// assert!(matches!(backend, LLMBackend::OpenAI));
///
/// let err = LLMBackend::from_str("invalid").unwrap_err();
/// assert!(err.to_string().contains("Unknown LLM backend"));
/// ```
impl std::str::FromStr for LLMBackend {
    type Err = LLMError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(LLMBackend::OpenAI),
            "anthropic" => Ok(LLMBackend::Anthropic),
            "ollama" => Ok(LLMBackend::Ollama),
            "deepseek" => Ok(LLMBackend::DeepSeek),
            "xai" => Ok(LLMBackend::XAI),
            "phind" => Ok(LLMBackend::Phind),
            "google" => Ok(LLMBackend::Google),
            "groq" => Ok(LLMBackend::Groq),
            "azure-openai" => Ok(LLMBackend::AzureOpenAI),
            "elevenlabs" => Ok(LLMBackend::ElevenLabs),
            "cohere" => Ok(LLMBackend::Cohere),
            "mistral" => Ok(LLMBackend::Mistral),
            "openrouter" => Ok(LLMBackend::OpenRouter),
            "huggingface" => Ok(LLMBackend::HuggingFace),
            _ => Err(LLMError::InvalidRequest(format!(
                "Unknown LLM backend: {s}"
            ))),
        }
    }
}

/// Builder for configuring and instantiating LLM providers.
///
/// Provides a fluent interface for setting various configuration options
/// like model selection, API keys, generation parameters, etc.
#[derive(Default)]
pub struct LLMBuilder {
    /// Selected backend provider
    backend: Option<LLMBackend>,
    /// API key for authentication with the provider
    api_key: Option<String>,
    /// Base URL for API requests (primarily for self-hosted instances)
    base_url: Option<String>,
    /// Model identifier/name to use
    model: Option<String>,
    /// Maximum tokens to generate in responses
    max_tokens: Option<u32>,
    /// Temperature parameter for controlling response randomness (0.0-1.0)
    temperature: Option<f32>,
    /// System prompt/context to guide model behavior
    system: Option<String>,
    /// Request timeout duration in seconds
    timeout_seconds: Option<u64>,
    /// Top-p (nucleus) sampling parameter
    top_p: Option<f32>,
    /// Top-k sampling parameter
    top_k: Option<u32>,
    /// Format specification for embedding outputs
    embedding_encoding_format: Option<String>,
    /// Vector dimensions for embedding outputs
    embedding_dimensions: Option<u32>,
    /// Optional validation function for response content
    validator: Option<Box<ValidatorFn>>,
    /// Number of retry attempts when validation fails
    validator_attempts: usize,
    /// Function tools
    tools: Option<Vec<Tool>>,
    /// Tool choice
    tool_choice: Option<ToolChoice>,
    /// Enable parallel tool use
    enable_parallel_tool_use: Option<bool>,
    /// Increase response consistency by normalizing output, e.g. for streaming tool calls
    normalize_response: Option<bool>,
    /// Enable reasoning
    reasoning: Option<bool>,
    /// Enable reasoning effort
    reasoning_effort: Option<String>,
    /// reasoning_budget_tokens
    reasoning_budget_tokens: Option<u32>,
    /// JSON schema for structured output
    json_schema: Option<StructuredOutputFormat>,
    /// API Version
    api_version: Option<String>,
    /// Deployment Id
    deployment_id: Option<String>,
    /// Voice
    voice: Option<String>,
    /// ExtraBody
    extra_body: Option<serde_json::Value>,
    /// Search parameters for providers that support search functionality
    xai_search_mode: Option<String>,
    /// XAI search source type
    xai_search_source_type: Option<String>,
    /// XAI search excluded websites
    xai_search_excluded_websites: Option<Vec<String>>,
    /// XAI search max results
    xai_search_max_results: Option<u32>,
    /// XAI search from date
    xai_search_from_date: Option<String>,
    /// XAI search to date
    xai_search_to_date: Option<String>,
    /// Memory provider for conversation history (optional)
    memory: Option<Box<dyn MemoryProvider>>,
    /// Use web search
    openai_enable_web_search: Option<bool>,
    /// OpenAI web search context
    openai_web_search_context_size: Option<String>,
    /// OpenAI web search user location type
    openai_web_search_user_location_type: Option<String>,
    /// OpenAI web search user location approximate country
    openai_web_search_user_location_approximate_country: Option<String>,
    /// OpenAI web search user location approximate city
    openai_web_search_user_location_approximate_city: Option<String>,
    /// OpenAI web search user location approximate region
    openai_web_search_user_location_approximate_region: Option<String>,
    /// Resilience: enable retry/backoff wrapper
    resilient_enable: Option<bool>,
    /// Resilience: max attempts
    resilient_attempts: Option<usize>,
    /// Resilience: base and max delay in ms
    resilient_base_delay_ms: Option<u64>,
    resilient_max_delay_ms: Option<u64>,
    /// Resilience: jitter toggle
    resilient_jitter: Option<bool>,
}

impl LLMBuilder {
    /// Creates a new empty builder instance with default values.
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    /// Sets the backend provider to use.
    pub fn backend(mut self, backend: LLMBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Sets the API key for authentication.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL for API requests.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the model identifier to use.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the temperature for controlling response randomness (0.0-1.0).
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the system prompt/context.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the reasoning flag.
    pub fn reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort.to_string());
        self
    }

    /// Sets the reasoning flag.
    pub fn reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Sets the reasoning budget tokens.
    pub fn reasoning_budget_tokens(mut self, reasoning_budget_tokens: u32) -> Self {
        self.reasoning_budget_tokens = Some(reasoning_budget_tokens);
        self
    }

    /// Sets the request timeout in seconds.
    pub fn timeout_seconds(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = Some(timeout_seconds);
        self
    }

    /// Enables or disables streaming responses, stub kept for compatibility.
    ///
    /// # Deprecation
    /// This method is deprecated and will be removed in a future release.
    /// Streaming is defined by the function called (e.g. `chat_stream`).
    #[deprecated(
        note = "This method is deprecated and will be removed in a future release. Streaming is defined by the function used (e.g. `chat_stream`)"
    )]
    pub fn stream(self, _stream: bool) -> Self {
        self
    }

    /// Sets the request timeout in seconds.
    pub fn normalize_response(mut self, normalize_response: bool) -> Self {
        self.normalize_response = Some(normalize_response);
        self
    }

    /// Sets the top-p (nucleus) sampling parameter.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the top-k sampling parameter.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Sets the encoding format for embeddings.
    pub fn embedding_encoding_format(
        mut self,
        embedding_encoding_format: impl Into<String>,
    ) -> Self {
        self.embedding_encoding_format = Some(embedding_encoding_format.into());
        self
    }

    /// Sets the dimensions for embeddings.
    pub fn embedding_dimensions(mut self, embedding_dimensions: u32) -> Self {
        self.embedding_dimensions = Some(embedding_dimensions);
        self
    }

    /// Sets the JSON schema for structured output.
    pub fn schema(mut self, schema: impl Into<StructuredOutputFormat>) -> Self {
        self.json_schema = Some(schema.into());
        self
    }

    /// Sets a validation function to verify LLM responses.
    ///
    /// # Arguments
    ///
    /// * `f` - Function that takes a response string and returns Ok(()) if valid, or Err with error message if invalid
    pub fn validator<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Result<(), String> + Send + Sync + 'static,
    {
        self.validator = Some(Box::new(f));
        self
    }

    /// Sets the number of retry attempts for validation failures.
    ///
    /// # Arguments
    ///
    /// * `attempts` - Maximum number of times to retry generating a valid response
    pub fn validator_attempts(mut self, attempts: usize) -> Self {
        self.validator_attempts = attempts;
        self
    }

    /// Adds a function tool to the builder
    pub fn function(mut self, function_builder: FunctionBuilder) -> Self {
        if self.tools.is_none() {
            self.tools = Some(Vec::new());
        }
        if let Some(tools) = &mut self.tools {
            tools.push(function_builder.build());
        }
        self
    }

    /// Enable parallel tool use
    pub fn enable_parallel_tool_use(mut self, enable: bool) -> Self {
        self.enable_parallel_tool_use = Some(enable);
        self
    }

    /// Set tool choice.  Note that if the choice is given as Tool(name), and that
    /// tool isn't available, the builder will fail.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Explicitly disable the use of tools, even if they are provided.
    pub fn disable_tools(mut self) -> Self {
        self.tool_choice = Some(ToolChoice::None);
        self
    }

    /// Set the API version.
    pub fn api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = Some(api_version.into());
        self
    }

    /// Set the deployment id. Used in Azure OpenAI.
    pub fn deployment_id(mut self, deployment_id: impl Into<String>) -> Self {
        self.deployment_id = Some(deployment_id.into());
        self
    }

    /// Set the voice.
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Set the extra body.
    pub fn extra_body(mut self, extra_body: impl serde::Serialize) -> Self {
        let value = serde_json::to_value(extra_body).ok();
        self.extra_body = value;
        self
    }

    /// Enable web search
    pub fn openai_enable_web_search(mut self, enable: bool) -> Self {
        self.openai_enable_web_search = Some(enable);
        self
    }

    /// Set the web search context
    pub fn openai_web_search_context_size(mut self, context_size: impl Into<String>) -> Self {
        self.openai_web_search_context_size = Some(context_size.into());
        self
    }

    /// Set the web search user location type
    pub fn openai_web_search_user_location_type(
        mut self,
        location_type: impl Into<String>,
    ) -> Self {
        self.openai_web_search_user_location_type = Some(location_type.into());
        self
    }

    /// Set the web search user location approximate country
    pub fn openai_web_search_user_location_approximate_country(
        mut self,
        country: impl Into<String>,
    ) -> Self {
        self.openai_web_search_user_location_approximate_country = Some(country.into());
        self
    }

    /// Set the web search user location approximate city
    pub fn openai_web_search_user_location_approximate_city(
        mut self,
        city: impl Into<String>,
    ) -> Self {
        self.openai_web_search_user_location_approximate_city = Some(city.into());
        self
    }

    /// Set the web search user location approximate region
    pub fn openai_web_search_user_location_approximate_region(
        mut self,
        region: impl Into<String>,
    ) -> Self {
        self.openai_web_search_user_location_approximate_region = Some(region.into());
        self
    }

    /// Enables or disables the resilience wrapper (retry/backoff).
    pub fn resilient(mut self, enable: bool) -> Self {
        self.resilient_enable = Some(enable);
        self
    }

    /// Sets the maximum number of attempts for resilience (including the first try).
    pub fn resilient_attempts(mut self, attempts: usize) -> Self {
        self.resilient_attempts = Some(attempts);
        self
    }

    /// Sets the backoff bounds in milliseconds for resilience.
    pub fn resilient_backoff(mut self, base_delay_ms: u64, max_delay_ms: u64) -> Self {
        self.resilient_base_delay_ms = Some(base_delay_ms);
        self.resilient_max_delay_ms = Some(max_delay_ms);
        self
    }

    /// Enables or disables jitter for backoff delays.
    pub fn resilient_jitter(mut self, jitter: bool) -> Self {
        self.resilient_jitter = Some(jitter);
        self
    }

    #[deprecated(note = "Renamed to `xai_search_mode`.")]
    pub fn search_mode(self, mode: impl Into<String>) -> Self {
        self.xai_search_mode(mode)
    }

    /// Sets the search mode for search-enabled providers.
    pub fn xai_search_mode(mut self, mode: impl Into<String>) -> Self {
        self.xai_search_mode = Some(mode.into());
        self
    }

    /// Adds a search source with optional excluded websites.
    pub fn xai_search_source(
        mut self,
        source_type: impl Into<String>,
        excluded_websites: Option<Vec<String>>,
    ) -> Self {
        self.xai_search_source_type = Some(source_type.into());
        self.xai_search_excluded_websites = excluded_websites;
        self
    }

    /// Sets the maximum number of search results.
    pub fn xai_max_search_results(mut self, max: u32) -> Self {
        self.xai_search_max_results = Some(max);
        self
    }

    /// Sets the date range for search results.
    pub fn xai_search_date_range(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.xai_search_from_date = Some(from.into());
        self.xai_search_to_date = Some(to.into());
        self
    }

    /// Sets the start date for search results (format: "YYYY-MM-DD").
    pub fn xai_search_from_date(mut self, date: impl Into<String>) -> Self {
        self.xai_search_from_date = Some(date.into());
        self
    }

    /// Sets the end date for search results (format: "YYYY-MM-DD").
    pub fn xai_search_to_date(mut self, date: impl Into<String>) -> Self {
        self.xai_search_to_date = Some(date.into());
        self
    }

    /// Sets a custom memory provider for storing conversation history.
    ///
    /// # Arguments
    ///
    /// * `memory` - A boxed memory provider implementing the MemoryProvider trait
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llm::builder::{LLMBuilder, LLMBackend};
    /// use llm::memory::SlidingWindowMemory;
    ///
    /// let memory = SlidingWindowMemory::new(10);
    /// let builder = LLMBuilder::new()
    ///     .backend(LLMBackend::OpenAI)
    ///     .memory(memory);
    /// ```
    pub fn memory(mut self, memory: impl MemoryProvider + 'static) -> Self {
        self.memory = Some(Box::new(memory));
        self
    }

    /// Sets a sliding window memory instance directly (convenience method).
    ///
    /// # Arguments
    ///
    /// * `memory` - A SlidingWindowMemory instance
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llm::builder::{LLMBuilder, LLMBackend};
    /// use llm::memory::SlidingWindowMemory;
    ///
    /// let memory = SlidingWindowMemory::new(10);
    /// let builder = LLMBuilder::new()
    ///     .backend(LLMBackend::OpenAI)
    ///     .sliding_memory(memory);
    /// ```
    pub fn sliding_memory(mut self, memory: SlidingWindowMemory) -> Self {
        self.memory = Some(Box::new(memory));
        self
    }

    /// Sets up a sliding window memory with the specified window size.
    ///
    /// This is a convenience method for creating a SlidingWindowMemory instance.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Maximum number of messages to keep in memory
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llm::builder::{LLMBuilder, LLMBackend};
    ///
    /// let builder = LLMBuilder::new()
    ///     .backend(LLMBackend::OpenAI)
    ///     .sliding_window_memory(5); // Keep last 5 messages
    /// ```
    pub fn sliding_window_memory(mut self, window_size: usize) -> Self {
        self.memory = Some(Box::new(SlidingWindowMemory::new(window_size)));
        self
    }

    /// Sets up a sliding window memory with specified trim strategy.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Maximum number of messages to keep in memory
    /// * `strategy` - How to handle overflow when window is full
    ///
    /// # Examples
    ///
    /// ```rust
    /// use llm::builder::{LLMBuilder, LLMBackend};
    /// use llm::memory::TrimStrategy;
    ///
    /// let builder = LLMBuilder::new()
    ///     .backend(LLMBackend::OpenAI)
    ///     .sliding_window_with_strategy(5, TrimStrategy::Summarize);
    /// ```
    pub fn sliding_window_with_strategy(
        mut self,
        window_size: usize,
        strategy: TrimStrategy,
    ) -> Self {
        self.memory = Some(Box::new(SlidingWindowMemory::with_strategy(
            window_size,
            strategy,
        )));
        self
    }

    /// Builds and returns a configured LLM provider instance.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No backend is specified
    /// - Required backend feature is not enabled
    /// - Required configuration like API keys are missing
    pub fn build(self) -> Result<Box<dyn LLMProvider>, LLMError> {
        log::debug!(
            "Building LLM provider. backend={:?} model={:?} tools={} tool_choice={:?} temp={:?} enable_web_search={:?} web_search_context={:?} web_search_user_location_type={:?} web_search_user_location_approximate_country={:?} web_search_user_location_approximate_city={:?} web_search_user_location_approximate_region={:?}",
            self.backend,
            self.model,
            self.tools.as_ref().map(|v| v.len()).unwrap_or(0),
            self.tool_choice,
            self.temperature,
            self.openai_enable_web_search,
            self.openai_web_search_context_size,
            self.openai_web_search_user_location_type,
            self.openai_web_search_user_location_approximate_country,
            self.openai_web_search_user_location_approximate_city,
            self.openai_web_search_user_location_approximate_region,
        );
        let (tools, tool_choice) = self.validate_tool_config()?;
        let backend = self
            .backend
            .ok_or_else(|| LLMError::InvalidRequest("No backend specified".to_string()))?;

        #[allow(unused_variables)]
        let provider: Box<dyn LLMProvider> = match backend {
            LLMBackend::OpenAI => {
                #[cfg(not(feature = "openai"))]
                return Err(LLMError::InvalidRequest(
                    "OpenAI feature not enabled".to_string(),
                ));
                #[cfg(feature = "openai")]
                {
                    let key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for OpenAI".to_string())
                    })?;
                    Box::new(crate::backends::openai::OpenAI::new(
                        key,
                        self.base_url,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        self.embedding_encoding_format,
                        self.embedding_dimensions,
                        tools,
                        tool_choice,
                        self.normalize_response,
                        self.reasoning_effort,
                        self.json_schema,
                        self.voice,
                        self.extra_body,
                        self.openai_enable_web_search,
                        self.openai_web_search_context_size,
                        self.openai_web_search_user_location_type,
                        self.openai_web_search_user_location_approximate_country,
                        self.openai_web_search_user_location_approximate_city,
                        self.openai_web_search_user_location_approximate_region,
                    )?)
                }
            }
            LLMBackend::ElevenLabs => {
                #[cfg(not(feature = "elevenlabs"))]
                return Err(LLMError::InvalidRequest(
                    "ElevenLabs feature not enabled".to_string(),
                ));
                #[cfg(feature = "elevenlabs")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for ElevenLabs".to_string())
                    })?;
                    let elevenlabs = crate::backends::elevenlabs::ElevenLabs::new(
                        api_key,
                        self.model.unwrap_or("eleven_multilingual_v2".to_string()),
                        "https://api.elevenlabs.io/v1".to_string(),
                        self.timeout_seconds,
                        self.voice,
                    );
                    Box::new(elevenlabs)
                }
            }
            LLMBackend::Anthropic => {
                #[cfg(not(feature = "anthropic"))]
                return Err(LLMError::InvalidRequest(
                    "Anthropic feature not enabled".to_string(),
                ));
                #[cfg(feature = "anthropic")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for Anthropic".to_string())
                    })?;
                    let anthro = crate::backends::anthropic::Anthropic::new(
                        api_key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        tools,
                        self.tool_choice,
                        self.reasoning,
                        self.reasoning_budget_tokens,
                    );
                    Box::new(anthro)
                }
            }
            LLMBackend::Ollama => {
                #[cfg(not(feature = "ollama"))]
                return Err(LLMError::InvalidRequest(
                    "Ollama feature not enabled".to_string(),
                ));
                #[cfg(feature = "ollama")]
                {
                    let url = self
                        .base_url
                        .unwrap_or("http://localhost:11434".to_string());
                    let ollama = crate::backends::ollama::Ollama::new(
                        url,
                        self.api_key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        self.json_schema,
                        tools,
                    );
                    Box::new(ollama)
                }
            }
            LLMBackend::DeepSeek => {
                #[cfg(not(feature = "deepseek"))]
                return Err(LLMError::InvalidRequest(
                    "DeepSeek feature not enabled".to_string(),
                ));

                #[cfg(feature = "deepseek")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for DeepSeek".to_string())
                    })?;
                    let deepseek = crate::backends::deepseek::DeepSeek::new(
                        api_key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                    );
                    Box::new(deepseek)
                }
            }
            LLMBackend::XAI => {
                #[cfg(not(feature = "xai"))]
                return Err(LLMError::InvalidRequest(
                    "XAI feature not enabled".to_string(),
                ));

                #[cfg(feature = "xai")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for XAI".to_string())
                    })?;

                    let xai = crate::backends::xai::XAI::new(
                        api_key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        self.embedding_encoding_format,
                        self.embedding_dimensions,
                        self.json_schema,
                        self.xai_search_mode,
                        self.xai_search_source_type,
                        self.xai_search_excluded_websites,
                        self.xai_search_max_results,
                        self.xai_search_from_date,
                        self.xai_search_to_date,
                    );
                    Box::new(xai)
                }
            }
            LLMBackend::Phind => {
                #[cfg(not(feature = "phind"))]
                return Err(LLMError::InvalidRequest(
                    "Phind feature not enabled".to_string(),
                ));

                #[cfg(feature = "phind")]
                {
                    let phind = crate::backends::phind::Phind::new(
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                    );
                    Box::new(phind)
                }
            }
            LLMBackend::Google => {
                #[cfg(not(feature = "google"))]
                return Err(LLMError::InvalidRequest(
                    "Google feature not enabled".to_string(),
                ));

                #[cfg(feature = "google")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for Google".to_string())
                    })?;

                    let google = crate::backends::google::Google::new(
                        api_key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        self.json_schema,
                        tools,
                    );
                    Box::new(google)
                }
            }
            LLMBackend::Groq => {
                #[cfg(not(feature = "groq"))]
                return Err(LLMError::InvalidRequest(
                    "Groq feature not enabled".to_string(),
                ));

                #[cfg(feature = "groq")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for Groq".to_string())
                    })?;

                    let groq = crate::backends::groq::Groq::with_config(
                        api_key,
                        self.base_url,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        self.tools,
                        self.tool_choice,
                        self.extra_body,
                        None, // embedding_encoding_format
                        None, // embedding_dimensions
                        None, // reasoning_effort
                        self.json_schema,
                        self.enable_parallel_tool_use,
                        self.normalize_response,
                    );
                    Box::new(groq)
                }
            }
            LLMBackend::OpenRouter => {
                #[cfg(not(feature = "openrouter"))]
                return Err(LLMError::InvalidRequest(
                    "OpenRouter feature not enabled".to_string(),
                ));

                #[cfg(feature = "openrouter")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for OpenRouter".to_string())
                    })?;

                    let openrouter = crate::backends::openrouter::OpenRouter::with_config(
                        api_key,
                        self.base_url,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        self.tools,
                        self.tool_choice,
                        self.extra_body,
                        None, // embedding_encoding_format
                        None, // embedding_dimensions
                        None, // reasoning_effort
                        self.json_schema,
                        self.enable_parallel_tool_use,
                        self.normalize_response,
                    );
                    Box::new(openrouter)
                }
            }
            LLMBackend::Cohere => {
                #[cfg(not(feature = "cohere"))]
                return Err(LLMError::InvalidRequest(
                    "Cohere feature not enabled".to_string(),
                ));

                #[cfg(feature = "cohere")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for Cohere".to_string())
                    })?;
                    let cohere = crate::backends::cohere::Cohere::new(
                        api_key,
                        self.base_url,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        tools,
                        self.tool_choice,
                        self.reasoning_effort,
                        self.json_schema,
                        None,
                        self.extra_body,
                        self.enable_parallel_tool_use,
                        self.normalize_response,
                        self.embedding_encoding_format,
                        self.embedding_dimensions,
                    );
                    Box::new(cohere)
                }
            }
            LLMBackend::HuggingFace => {
                #[cfg(not(feature = "huggingface"))]
                return Err(LLMError::InvalidRequest(
                    "huggingface feature not enabled".to_string(),
                ));

                #[cfg(feature = "huggingface")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest(
                            "No API key provided for HuggingFace Inference Providers".to_string(),
                        )
                    })?;

                    let llm = crate::backends::huggingface::HuggingFace::with_config(
                        api_key,
                        self.base_url,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        self.tools,
                        self.tool_choice,
                        self.extra_body,
                        None, // embedding_encoding_format
                        None, // embedding_dimensions
                        None, // reasoning_effort
                        self.json_schema,
                        self.enable_parallel_tool_use,
                        self.normalize_response,
                    );
                    Box::new(llm)
                }
            }
            LLMBackend::Mistral => {
                #[cfg(not(feature = "mistral"))]
                return Err(LLMError::InvalidRequest(
                    "Mistral feature not enabled".to_string(),
                ));
                #[cfg(feature = "mistral")]
                {
                    let api_key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for Mistral".to_string())
                    })?;
                    let mistral = crate::backends::mistral::Mistral::with_config(
                        api_key,
                        self.base_url,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        tools,
                        tool_choice,
                        self.extra_body,
                        self.embedding_encoding_format,
                        self.embedding_dimensions,
                        self.reasoning_effort,
                        self.json_schema,
                        self.enable_parallel_tool_use,
                        self.normalize_response,
                    );
                    Box::new(mistral)
                }
            }
            LLMBackend::AzureOpenAI => {
                #[cfg(not(feature = "azure_openai"))]
                return Err(LLMError::InvalidRequest(
                    "OpenAI feature not enabled".to_string(),
                ));
                #[cfg(feature = "azure_openai")]
                {
                    let endpoint = self.base_url.ok_or_else(|| {
                        LLMError::InvalidRequest("No API endpoint provided for Azure OpenAI".into())
                    })?;
                    let key = self.api_key.ok_or_else(|| {
                        LLMError::InvalidRequest("No API key provided for Azure OpenAI".to_string())
                    })?;
                    let api_version = self.api_version.ok_or_else(|| {
                        LLMError::InvalidRequest(
                            "No API version provided for Azure OpenAI".to_string(),
                        )
                    })?;
                    let deployment = self.deployment_id.ok_or_else(|| {
                        LLMError::InvalidRequest(
                            "No deployment ID provided for Azure OpenAI".into(),
                        )
                    })?;
                    Box::new(crate::backends::azure_openai::AzureOpenAI::new(
                        key,
                        api_version,
                        deployment,
                        endpoint,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        self.embedding_encoding_format,
                        self.embedding_dimensions,
                        tools,
                        tool_choice,
                        self.reasoning_effort,
                        self.json_schema,
                    ))
                }
            }
            LLMBackend::AwsBedrock => {
                #[cfg(not(feature = "bedrock"))]
                return Err(LLMError::InvalidRequest(
                    "AWS Bedrock feature not enabled".to_string(),
                ));
                #[cfg(feature = "bedrock")]
                {
                    let region = self.base_url.ok_or_else(|| {
                        LLMError::InvalidRequest("No region provided for AWS Bedrock".into())
                    })?;
                    Box::new(crate::backends::aws::BedrockBackend::new(
                        region,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.top_p,
                        self.top_k,
                        tools,
                        tool_choice,
                        self.reasoning_effort,
                        self.json_schema,
                    )?)
                }
            }
        };

        #[allow(unreachable_code)]
        let mut final_provider: Box<dyn LLMProvider> = if let Some(validator) = self.validator {
            Box::new(crate::validated_llm::ValidatedLLM::new(
                provider,
                validator,
                self.validator_attempts,
            ))
        } else {
            provider
        };

        // Wrap with resilience retry/backoff if enabled
        if self.resilient_enable.unwrap_or(false) {
            let mut cfg = crate::resilient_llm::ResilienceConfig::defaults();
            if let Some(attempts) = self.resilient_attempts {
                cfg.max_attempts = attempts;
            }
            if let Some(base) = self.resilient_base_delay_ms {
                cfg.base_delay_ms = base;
            }
            if let Some(maxd) = self.resilient_max_delay_ms {
                cfg.max_delay_ms = maxd;
            }
            if let Some(j) = self.resilient_jitter {
                cfg.jitter = j;
            }
            final_provider = Box::new(crate::resilient_llm::ResilientLLM::new(final_provider, cfg));
        }

        // Wrap with memory capabilities if memory is configured
        if let Some(memory) = self.memory {
            let memory_arc = Arc::new(RwLock::new(memory));
            let provider_arc = Arc::from(final_provider);
            final_provider = Box::new(ChatWithMemory::new(
                provider_arc,
                memory_arc,
                None,
                Vec::new(),
                None,
            ));
        }
        Ok(final_provider)
    }

    // Validate that tool configuration is consistent and valid
    fn validate_tool_config(&self) -> Result<(Option<Vec<Tool>>, Option<ToolChoice>), LLMError> {
        match self.tool_choice {
            Some(ToolChoice::Tool(ref name)) => {
                match self.tools.clone().map(|tools| tools.iter().any(|tool| tool.function.name == *name)) {
                    Some(true) => Ok((self.tools.clone(), self.tool_choice.clone())),
                    _ => Err(LLMError::ToolConfigError(format!("Tool({name}) cannot be tool choice: no tool with name {name} found.  Did you forget to add it with .function?"))),
                }
            }
            Some(_) if self.tools.is_none() => Err(LLMError::ToolConfigError(
                "Tool choice cannot be set without tools configured".to_string(),
            )),
            _ => Ok((self.tools.clone(), self.tool_choice.clone())),
        }
    }
}

/// Builder for function parameters
pub struct ParamBuilder {
    name: String,
    property_type: String,
    description: String,
    items: Option<Box<ParameterProperty>>,
    enum_list: Option<Vec<String>>,
}

impl ParamBuilder {
    /// Creates a new parameter builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            property_type: "string".to_string(),
            description: String::new(),
            items: None,
            enum_list: None,
        }
    }

    /// Sets the parameter type
    pub fn type_of(mut self, type_str: impl Into<String>) -> Self {
        self.property_type = type_str.into();
        self
    }

    /// Sets the parameter description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Sets the array item type for array parameters
    pub fn items(mut self, item_property: ParameterProperty) -> Self {
        self.items = Some(Box::new(item_property));
        self
    }

    /// Sets the enum values for enum parameters
    pub fn enum_values(mut self, values: Vec<String>) -> Self {
        self.enum_list = Some(values);
        self
    }

    /// Builds the parameter property
    fn build(self) -> (String, ParameterProperty) {
        (
            self.name,
            ParameterProperty {
                property_type: self.property_type,
                description: self.description,
                items: self.items,
                enum_list: self.enum_list,
            },
        )
    }
}

/// Builder for function tools
pub struct FunctionBuilder {
    name: String,
    description: String,
    parameters: Vec<ParamBuilder>,
    required: Vec<String>,
    raw_schema: Option<serde_json::Value>,
}

impl FunctionBuilder {
    /// Creates a new function builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            parameters: Vec::new(),
            required: Vec::new(),
            raw_schema: None,
        }
    }

    /// Sets the function description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Adds a parameter to the function
    pub fn param(mut self, param: ParamBuilder) -> Self {
        self.parameters.push(param);
        self
    }

    /// Marks parameters as required
    pub fn required(mut self, param_names: Vec<String>) -> Self {
        self.required = param_names;
        self
    }

    /// Provides a full JSON Schema for the parameters.  Using this method
    /// bypasses the DSL and allows arbitrary complex schemas (nested arrays,
    /// objects, oneOf, etc.).
    pub fn json_schema(mut self, schema: serde_json::Value) -> Self {
        self.raw_schema = Some(schema);
        self
    }

    /// Builds the function tool
    fn build(self) -> Tool {
        let parameters_value = if let Some(schema) = self.raw_schema {
            schema
        } else {
            let mut properties = HashMap::new();
            for param in self.parameters {
                let (name, prop) = param.build();
                properties.insert(name, prop);
            }
            serde_json::to_value(ParametersSchema {
                schema_type: "object".to_string(),
                properties,
                required: self.required,
            })
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
        };
        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: self.name,
                description: self.description,
                parameters: parameters_value,
            },
        }
    }
}
