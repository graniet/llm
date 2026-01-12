use std::sync::Arc;

use tokio::sync::RwLock;

use crate::{
    error::LLMError,
    memory::ChatWithMemoryConfig,
    resilient_llm::{ResilienceConfig, ResilientLLM},
    validated_llm::ValidatedLLM,
    LLMProvider,
};

use super::super::state::BuilderState;

pub(super) fn wrap_with_validator(
    state: &mut BuilderState,
    provider: Box<dyn LLMProvider>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let Some(validator) = state.validator.take() else {
        return Ok(provider);
    };
    if state.validator_attempts == 0 {
        return Err(LLMError::InvalidRequest(
            "validator_attempts must be greater than 0".to_string(),
        ));
    }
    Ok(Box::new(ValidatedLLM::new(
        provider,
        validator,
        state.validator_attempts,
    )))
}

pub(super) fn wrap_with_resilience(
    state: &mut BuilderState,
    provider: Box<dyn LLMProvider>,
) -> Box<dyn LLMProvider> {
    if !state.resilient_enable.unwrap_or(false) {
        return provider;
    }

    let mut cfg = ResilienceConfig::defaults();
    if let Some(attempts) = state.resilient_attempts {
        cfg.max_attempts = attempts;
    }
    if let Some(base) = state.resilient_base_delay_ms {
        cfg.base_delay_ms = base;
    }
    if let Some(maxd) = state.resilient_max_delay_ms {
        cfg.max_delay_ms = maxd;
    }
    if let Some(jitter) = state.resilient_jitter {
        cfg.jitter = jitter;
    }
    Box::new(ResilientLLM::new(provider, cfg))
}

pub(super) fn wrap_with_memory(
    state: &mut BuilderState,
    provider: Box<dyn LLMProvider>,
) -> Box<dyn LLMProvider> {
    let Some(memory) = state.memory.take() else {
        return provider;
    };

    let memory_arc = Arc::new(RwLock::new(memory));
    let provider_arc = Arc::from(provider);
    let config = ChatWithMemoryConfig::new(provider_arc, memory_arc);
    Box::new(crate::memory::ChatWithMemory::with_config(config))
}
