//! Builder module for configuring and instantiating LLM providers.
//!
//! This module provides a flexible builder pattern for creating and configuring
//! LLM (Large Language Model) provider instances with various settings and options.

use serde_json::Value;

use crate::{
    chat::{FunctionTool, ParameterProperty, ParametersSchema, ReasoningEffort, Tool},
    error::LLMError,
    LLMProvider,
};
use std::collections::HashMap;

/// A function type for validating LLM provider outputs.
/// Takes a response string and returns Ok(()) if valid, or Err with an error message if invalid.
pub type ValidatorFn = dyn Fn(&str) -> Result<(), String> + Send + Sync + 'static;

/// Supported LLM backend providers.
#[derive(Debug, Clone)]
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
    /// Whether to enable streaming responses
    stream: Option<bool>,
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
    /// Enable reasoning
    reasoning: Option<bool>,
    /// Enable reasoning effort
    reasoning_effort: Option<String>,
    /// reasoning_budget_tokens
    reasoning_budget_tokens: Option<u32>,
    /// JSON schema for structured output
    json_schema: Option<Value>,
}

impl LLMBuilder {
    /// Creates a new empty builder instance with default values.
    pub fn new() -> Self {
        Self::default()
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

    /// Enables or disables streaming responses.
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
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
    pub fn schema(mut self, schema: impl Into<Value>) -> Self {
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

    /// Builds and returns a configured LLM provider instance.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No backend is specified
    /// - Required backend feature is not enabled
    /// - Required configuration like API keys are missing
    pub fn build(self) -> Result<Box<dyn LLMProvider>, LLMError> {
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
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.stream,
                        self.top_p,
                        self.top_k,
                        self.embedding_encoding_format,
                        self.embedding_dimensions,
                        self.tools,
                        self.reasoning_effort,
                        self.json_schema,
                    ))
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
                        self.stream,
                        self.top_p,
                        self.top_k,
                        self.tools,
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
                        self.stream,
                        self.top_p,
                        self.top_k,
                        self.json_schema,
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
                        self.stream,
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
                        self.stream,
                        self.top_p,
                        self.top_k,
                        self.embedding_encoding_format,
                        self.embedding_dimensions,
                        self.json_schema,
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
                        self.stream,
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
                        self.stream,
                        self.top_p,
                        self.top_k,
                        self.json_schema,
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

                    let groq = crate::backends::groq::Groq::new(
                        api_key,
                        self.model,
                        self.max_tokens,
                        self.temperature,
                        self.timeout_seconds,
                        self.system,
                        self.stream,
                        self.top_p,
                        self.top_k,
                    );
                    Box::new(groq)
                }
            }
        };

        #[allow(unreachable_code)]
        if let Some(validator) = self.validator {
            Ok(Box::new(crate::validated_llm::ValidatedLLM::new(
                provider,
                validator,
                self.validator_attempts,
            )))
        } else {
            Ok(provider)
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
}

impl FunctionBuilder {
    /// Creates a new function builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            parameters: Vec::new(),
            required: Vec::new(),
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

    /// Builds the function tool
    fn build(self) -> Tool {
        let mut properties = HashMap::new();
        for param in self.parameters {
            let (name, prop) = param.build();
            properties.insert(name, prop);
        }

        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: self.name,
                description: self.description,
                parameters: ParametersSchema {
                    schema_type: "object".to_string(),
                    properties,
                    required: self.required,
                },
            },
        }
    }
}
