use crate::{error::LLMError, LLMProvider};

use super::super::helpers;
use crate::builder::state::BuilderState;

#[cfg(feature = "deepseek")]
pub(super) fn build_deepseek(state: &mut BuilderState) -> Result<Box<dyn LLMProvider>, LLMError> {
    let api_key = helpers::require_api_key(state, "DeepSeek")?;
    let timeout = helpers::timeout_or_default(state);

    let provider = crate::backends::deepseek::DeepSeek::new(
        api_key,
        state.model.take(),
        state.max_tokens,
        state.temperature,
        timeout,
        state.system.take(),
    );

    Ok(Box::new(provider))
}

#[cfg(not(feature = "deepseek"))]
pub(super) fn build_deepseek(_state: &mut BuilderState) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "DeepSeek feature not enabled".to_string(),
    ))
}
