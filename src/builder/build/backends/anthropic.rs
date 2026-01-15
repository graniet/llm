use crate::{
    builder::SystemPrompt,
    chat::{Tool, ToolChoice},
    error::LLMError,
    LLMProvider,
};

use super::super::helpers;
use crate::builder::state::BuilderState;

#[cfg(feature = "anthropic")]
pub(super) fn build_anthropic(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let api_key = helpers::require_api_key(state, "Anthropic")?;
    let timeout = helpers::timeout_or_default(state);

    // Convert String to SystemPrompt for Anthropic (supports structured prompts)
    let system_prompt = state.system.take().map(SystemPrompt::String);

    let provider = crate::backends::anthropic::Anthropic::new(
        api_key,
        state.model.take(),
        state.max_tokens,
        state.temperature,
        timeout,
        system_prompt,
        state.top_p,
        state.top_k,
        tools,
        tool_choice,
        state.reasoning,
        state.reasoning_budget_tokens,
    );

    Ok(Box::new(provider))
}

#[cfg(not(feature = "anthropic"))]
pub(super) fn build_anthropic(
    _state: &mut BuilderState,
    _tools: Option<Vec<Tool>>,
    _tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "Anthropic feature not enabled".to_string(),
    ))
}
