use crate::{chat::Tool, error::LLMError, LLMProvider};

use super::super::helpers;
use crate::builder::state::BuilderState;

#[cfg(feature = "google")]
use reqwest;

#[cfg(feature = "google")]
pub(super) fn build_google(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let api_key = helpers::require_api_key(state, "Google")?;
    let timeout = helpers::timeout_or_default(state);
    let vertex_ai_base_url = state.base_url.take();

    let provider = crate::backends::google::Google::with_client_and_vertex(
        reqwest::Client::builder()
            .build()
            .expect("Failed to build reqwest Client"),
        api_key,
        state.model.take(),
        state.max_tokens,
        state.temperature,
        timeout,
        state.system.take(),
        state.top_p,
        state.top_k,
        state.json_schema.take(),
        tools,
        state.google_service_tier.take(),
        vertex_ai_base_url,
    );

    Ok(Box::new(provider))
}

#[cfg(not(feature = "google"))]
pub(super) fn build_google(
    _state: &mut BuilderState,
    _tools: Option<Vec<Tool>>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "Google feature not enabled".to_string(),
    ))
}
