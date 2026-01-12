use crate::{
    chat::{Tool, ToolChoice},
    error::LLMError,
    LLMProvider,
};

use crate::builder::build::helpers;
use crate::builder::state::BuilderState;

#[cfg(feature = "huggingface")]
pub(super) fn build_huggingface(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let api_key = helpers::require_api_key(state, "HuggingFace Inference Providers")?;
    let timeout = helpers::timeout_or_default(state);
    let provider = crate::backends::huggingface::HuggingFace::with_config(
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
        state.extra_body.take(),
        None,
        None,
        None,
        state.json_schema.take(),
        state.enable_parallel_tool_use,
        state.normalize_response,
    );
    Ok(Box::new(provider))
}

#[cfg(not(feature = "huggingface"))]
pub(super) fn build_huggingface(
    _state: &mut BuilderState,
    _tools: Option<Vec<Tool>>,
    _tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "huggingface feature not enabled".to_string(),
    ))
}
