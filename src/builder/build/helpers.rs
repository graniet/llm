use secrecy::ExposeSecret;

use crate::{
    chat::{Tool, ToolChoice},
    error::LLMError,
};

use super::super::state::BuilderState;

const DEFAULT_TIMEOUT_SECONDS: u64 = 30;

pub(super) fn log_builder_state(state: &BuilderState) {
    log::debug!(
        "Building LLM provider. backend={:?} model={:?} tools={} tool_choice={:?} temp={:?} web_search={:?}",
        state.backend,
        state.model,
        state.tools.as_ref().map(|v| v.len()).unwrap_or(0),
        state.tool_choice,
        state.temperature,
        state.openai_enable_web_search,
    );
}

pub(super) fn validate_tool_config(
    state: &BuilderState,
) -> Result<(Option<Vec<Tool>>, Option<ToolChoice>), LLMError> {
    let tools = state.tools.clone();
    let choice = state.tool_choice.clone();

    match &choice {
        Some(ToolChoice::Tool(name)) => {
            let found = tools
                .as_ref()
                .map(|tools| tools.iter().any(|tool| tool.function.name == *name))
                .unwrap_or(false);
            if !found {
                return Err(LLMError::ToolConfigError(format!(
                    "Tool({name}) cannot be tool choice: no tool with name {name} found. Did you forget to add it with .function?"
                )));
            }
        }
        Some(_) if tools.is_none() => {
            return Err(LLMError::ToolConfigError(
                "Tool choice cannot be set without tools configured".to_string(),
            ));
        }
        _ => {}
    }

    Ok((tools, choice))
}

pub(super) fn require_api_key(
    state: &mut BuilderState,
    provider: &str,
) -> Result<String, LLMError> {
    let Some(key) = state.api_key.take() else {
        return Err(LLMError::InvalidRequest(format!(
            "No API key provided for {provider}"
        )));
    };
    Ok(key.expose_secret().to_string())
}

pub(super) fn optional_api_key(state: &mut BuilderState) -> Option<String> {
    state
        .api_key
        .take()
        .map(|key| key.expose_secret().to_string())
}

pub(super) fn timeout_or_default(state: &BuilderState) -> Option<u64> {
    Some(state.timeout_seconds.unwrap_or(DEFAULT_TIMEOUT_SECONDS))
}
