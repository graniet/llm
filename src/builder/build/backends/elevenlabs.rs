use crate::{error::LLMError, LLMProvider};

use super::super::helpers;
use crate::builder::state::BuilderState;

const DEFAULT_ELEVENLABS_MODEL: &str = "eleven_multilingual_v2";
const DEFAULT_ELEVENLABS_URL: &str = "https://api.elevenlabs.io/v1";

#[cfg(feature = "elevenlabs")]
pub(super) fn build_elevenlabs(state: &mut BuilderState) -> Result<Box<dyn LLMProvider>, LLMError> {
    let api_key = helpers::require_api_key(state, "ElevenLabs")?;
    let timeout = helpers::timeout_or_default(state);
    let model = state
        .model
        .take()
        .unwrap_or_else(|| DEFAULT_ELEVENLABS_MODEL.to_string());

    let provider = crate::backends::elevenlabs::ElevenLabs::new(
        api_key,
        model,
        DEFAULT_ELEVENLABS_URL.to_string(),
        timeout,
        state.voice.take(),
    );

    Ok(Box::new(provider))
}

#[cfg(not(feature = "elevenlabs"))]
pub(super) fn build_elevenlabs(
    _state: &mut BuilderState,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "ElevenLabs feature not enabled".to_string(),
    ))
}
