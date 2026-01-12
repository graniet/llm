use crate::{error::LLMError, LLMProvider};

use super::super::llm_builder::LLMBuilder;
use super::super::state::BuilderState;
use super::{backends, helpers, wrappers};

impl LLMBuilder {
    pub fn build(self) -> Result<Box<dyn LLMProvider>, LLMError> {
        self.state.build()
    }
}

impl BuilderState {
    pub(super) fn build(mut self) -> Result<Box<dyn LLMProvider>, LLMError> {
        helpers::log_builder_state(&self);
        let (tools, tool_choice) = helpers::validate_tool_config(&self)?;
        let backend = self
            .backend
            .take()
            .ok_or_else(|| LLMError::InvalidRequest("No backend specified".to_string()))?;

        let provider = backends::build_backend(&mut self, backend, tools, tool_choice)?;
        let provider = wrappers::wrap_with_validator(&mut self, provider)?;
        let provider = wrappers::wrap_with_resilience(&mut self, provider);
        let provider = wrappers::wrap_with_memory(&mut self, provider);
        Ok(provider)
    }
}
