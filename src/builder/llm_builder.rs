use secrecy::SecretString;

use crate::chat::ReasoningEffort;

use super::{backend::LLMBackend, state::BuilderState};

/// Builder for configuring and instantiating LLM providers.
pub struct LLMBuilder {
    pub(super) state: BuilderState,
}

impl Default for LLMBuilder {
    fn default() -> Self {
        Self {
            state: BuilderState::new(),
        }
    }
}

impl LLMBuilder {
    /// Creates a new empty builder instance with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the backend provider to use.
    pub fn backend(mut self, backend: LLMBackend) -> Self {
        self.state.backend = Some(backend);
        self
    }

    /// Sets the API key for authentication.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.state.api_key = Some(SecretString::new(key.into()));
        self
    }

    /// Sets the base URL for API requests.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.state.base_url = Some(url.into());
        self
    }

    /// Sets the model identifier to use.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.state.model = Some(model.into());
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.state.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the temperature for controlling response randomness (0.0-1.0).
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.state.temperature = Some(temperature);
        self
    }

    /// Sets the system prompt/context.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.state.system = Some(system.into());
        self
    }

    /// Sets the reasoning effort level.
    pub fn reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.state.reasoning_effort = Some(reasoning_effort.to_string());
        self
    }

    /// Sets the reasoning flag.
    pub fn reasoning(mut self, reasoning: bool) -> Self {
        self.state.reasoning = Some(reasoning);
        self
    }

    /// Sets the reasoning budget tokens.
    pub fn reasoning_budget_tokens(mut self, reasoning_budget_tokens: u32) -> Self {
        self.state.reasoning_budget_tokens = Some(reasoning_budget_tokens);
        self
    }

    /// Sets the request timeout in seconds.
    pub fn timeout_seconds(mut self, timeout_seconds: u64) -> Self {
        self.state.timeout_seconds = Some(timeout_seconds);
        self
    }

    /// No-op for compatibility (streaming handled by provider traits).
    pub fn stream(self, _stream: bool) -> Self {
        self
    }

    /// Sets whether to normalize responses.
    pub fn normalize_response(mut self, normalize_response: bool) -> Self {
        self.state.normalize_response = Some(normalize_response);
        self
    }

    /// Sets the top_p sampling parameter.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.state.top_p = Some(top_p);
        self
    }

    /// Sets the top_k sampling parameter.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.state.top_k = Some(top_k);
        self
    }
}
