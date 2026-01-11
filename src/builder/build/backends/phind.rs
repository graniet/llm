use crate::{error::LLMError, LLMProvider};

use super::super::helpers;
use crate::builder::state::BuilderState;

#[cfg(feature = "phind")]
pub(super) fn build_phind(state: &mut BuilderState) -> Result<Box<dyn LLMProvider>, LLMError> {
    let timeout = helpers::timeout_or_default(state);
    let provider = crate::backends::phind::Phind::new(
        state.model.take(),
        state.max_tokens,
        state.temperature,
        timeout,
        state.system.take(),
        state.top_p,
        state.top_k,
    );
    Ok(Box::new(provider))
}

#[cfg(not(feature = "phind"))]
pub(super) fn build_phind(_state: &mut BuilderState) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "Phind feature not enabled".to_string(),
    ))
}
