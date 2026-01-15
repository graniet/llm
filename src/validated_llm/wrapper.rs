use crate::{builder::ValidatorFn, LLMProvider};

/// A wrapper around an LLM provider that validates responses before returning them.
pub struct ValidatedLLM {
    pub(super) inner: Box<dyn LLMProvider>,
    pub(super) validator: Box<ValidatorFn>,
    pub(super) attempts: usize,
}

impl ValidatedLLM {
    /// Creates a new ValidatedLLM wrapper around an existing LLM provider.
    pub fn new(inner: Box<dyn LLMProvider>, validator: Box<ValidatorFn>, attempts: usize) -> Self {
        Self {
            inner,
            validator,
            attempts,
        }
    }

    pub(super) fn attempts(&self) -> usize {
        self.attempts
    }

    pub(super) fn validator(&self) -> &ValidatorFn {
        &self.validator
    }

    pub(super) fn inner(&self) -> &dyn LLMProvider {
        self.inner.as_ref()
    }
}
