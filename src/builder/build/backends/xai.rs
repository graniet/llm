use crate::{error::LLMError, LLMProvider};

use super::super::helpers;
use crate::builder::state::BuilderState;

#[cfg(feature = "xai")]
pub(super) fn build_xai(state: &mut BuilderState) -> Result<Box<dyn LLMProvider>, LLMError> {
    let api_key = helpers::require_api_key(state, "XAI")?;
    let timeout = helpers::timeout_or_default(state);

    let provider = crate::backends::xai::XAI::new(
        api_key,
        state.model.take(),
        state.max_tokens,
        state.temperature,
        timeout,
        state.system.take(),
        state.top_p,
        state.top_k,
        state.embedding_encoding_format.take(),
        state.embedding_dimensions,
        state.json_schema.take(),
        state.xai_search_mode.take(),
        state.xai_search_source_type.take(),
        state.xai_search_excluded_websites.take(),
        state.xai_search_max_results,
        state.xai_search_from_date.take(),
        state.xai_search_to_date.take(),
    );

    Ok(Box::new(provider))
}

#[cfg(not(feature = "xai"))]
pub(super) fn build_xai(_state: &mut BuilderState) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "XAI feature not enabled".to_string(),
    ))
}
