use crate::{
    chat::{Tool, ToolChoice},
    error::LLMError,
    LLMProvider,
};

use super::super::helpers;
use crate::builder::state::BuilderState;

#[cfg(feature = "openai")]
pub(super) fn build_openai(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let key = helpers::require_api_key(state, "OpenAI")?;
    let timeout = helpers::timeout_or_default(state);

    let provider = crate::backends::openai::OpenAI::new(
        key,
        state.base_url.take(),
        state.model.take(),
        state.max_tokens,
        state.temperature,
        timeout,
        state.system.take(),
        state.top_p,
        state.top_k,
        state.embedding_encoding_format.take(),
        state.embedding_dimensions,
        tools,
        tool_choice,
        state.normalize_response,
        state.reasoning_effort.take(),
        state.json_schema.take(),
        state.voice.take(),
        state.extra_body.take(),
        state.openai_enable_web_search,
        state.openai_web_search_context_size.take(),
        state.openai_web_search_user_location_type.take(),
        state
            .openai_web_search_user_location_approximate_country
            .take(),
        state
            .openai_web_search_user_location_approximate_city
            .take(),
        state
            .openai_web_search_user_location_approximate_region
            .take(),
    )?;

    Ok(Box::new(provider))
}

#[cfg(not(feature = "openai"))]
pub(super) fn build_openai(
    _state: &mut BuilderState,
    _tools: Option<Vec<Tool>>,
    _tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "OpenAI feature not enabled".to_string(),
    ))
}
