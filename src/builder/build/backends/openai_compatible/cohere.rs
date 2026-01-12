use crate::{
    chat::{Tool, ToolChoice},
    error::LLMError,
    LLMProvider,
};

use crate::builder::build::helpers;
use crate::builder::state::BuilderState;

#[cfg(feature = "cohere")]
pub(super) fn build_cohere(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let api_key = helpers::require_api_key(state, "Cohere")?;
    let timeout = helpers::timeout_or_default(state);
    let provider = crate::backends::cohere::Cohere::new(
        api_key,
        state.base_url.take(),
        state.model.take(),
        state.max_tokens,
        state.temperature,
        timeout,
        state.system.take(),
        state.top_p,
        state.top_k,
        tools,
        tool_choice,
        state.reasoning_effort.take(),
        state.json_schema.take(),
        None,
        state.extra_body.take(),
        state.enable_parallel_tool_use,
        state.normalize_response,
        state.embedding_encoding_format.take(),
        state.embedding_dimensions,
    );
    Ok(Box::new(provider))
}

#[cfg(not(feature = "cohere"))]
pub(super) fn build_cohere(
    _state: &mut BuilderState,
    _tools: Option<Vec<Tool>>,
    _tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "Cohere feature not enabled".to_string(),
    ))
}
