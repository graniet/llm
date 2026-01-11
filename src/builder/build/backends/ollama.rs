use crate::{chat::Tool, error::LLMError, LLMProvider};

use super::super::helpers;
use crate::builder::state::BuilderState;

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

#[cfg(feature = "ollama")]
pub(super) fn build_ollama(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let url = state
        .base_url
        .take()
        .unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());
    let timeout = helpers::timeout_or_default(state);

    let provider = crate::backends::ollama::Ollama::new(
        url,
        helpers::optional_api_key(state),
        state.model.take(),
        state.max_tokens,
        state.temperature,
        timeout,
        state.system.take(),
        state.top_p,
        state.top_k,
        state.json_schema.take(),
        tools,
    );

    Ok(Box::new(provider))
}

#[cfg(not(feature = "ollama"))]
pub(super) fn build_ollama(
    _state: &mut BuilderState,
    _tools: Option<Vec<Tool>>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "Ollama feature not enabled".to_string(),
    ))
}
