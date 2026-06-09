use crate::{error::LLMError, LLMProvider};

use super::super::helpers;
use crate::builder::state::BuilderState;

const DEFAULT_SIXTYDB_MODEL: &str = "60db Fast";
const DEFAULT_SIXTYDB_URL: &str = "https://api.60db.ai";

#[cfg(feature = "sixtydb")]
pub(super) fn build_sixtydb(state: &mut BuilderState) -> Result<Box<dyn LLMProvider>, LLMError> {
    let api_key = helpers::require_api_key(state, "60db")?;
    let timeout = helpers::timeout_or_default(state);
    let model = state
        .model
        .take()
        .unwrap_or_else(|| DEFAULT_SIXTYDB_MODEL.to_string());

    let provider = crate::backends::sixtydb::SixtyDb::new(
        api_key,
        model,
        DEFAULT_SIXTYDB_URL.to_string(),
        timeout,
        state.voice.take(),
    );

    Ok(Box::new(provider))
}

#[cfg(not(feature = "sixtydb"))]
pub(super) fn build_sixtydb(_state: &mut BuilderState) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "60db feature not enabled".to_string(),
    ))
}
