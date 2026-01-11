use super::llm_builder::LLMBuilder;

/// A function type for validating LLM provider outputs.
pub type ValidatorFn = dyn Fn(&str) -> Result<(), String> + Send + Sync + 'static;

impl LLMBuilder {
    /// Adds a validator function for responses.
    pub fn validator<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Result<(), String> + Send + Sync + 'static,
    {
        self.state.validator = Some(Box::new(f));
        self
    }

    /// Sets the number of validation attempts.
    pub fn validator_attempts(mut self, attempts: usize) -> Self {
        self.state.validator_attempts = attempts;
        self
    }
}
