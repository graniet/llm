use crate::{
    builder::SystemPrompt,
    chat::{Tool, ToolChoice},
    error::LLMError,
    LLMProvider,
};

use super::super::helpers;
use crate::builder::state::BuilderState;

#[cfg(feature = "vertex_ai")]
pub(super) fn build_vertex_ai(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let token = helpers::require_api_key(state, "Vertex AI")?;
    let base_url = state.base_url.take().ok_or_else(|| {
        LLMError::InvalidRequest(
            "No base_url provided for Vertex AI (expected https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location})".into(),
        )
    })?;
    let timeout = helpers::timeout_or_default(state);
    let system_prompt = state.system.take().map(SystemPrompt::String);

    let provider = crate::backends::vertex_ai::VertexAI::new(
        token,
        base_url,
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

#[cfg(not(feature = "vertex_ai"))]
pub(super) fn build_vertex_ai(
    _state: &mut BuilderState,
    _tools: Option<Vec<Tool>>,
    _tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "vertex_ai feature not enabled".to_string(),
    ))
}
