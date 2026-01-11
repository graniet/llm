use super::llm_builder::LLMBuilder;

impl LLMBuilder {
    /// Enable resilience retry/backoff wrapper.
    pub fn resilient(mut self, enable: bool) -> Self {
        self.state.resilient_enable = Some(enable);
        self
    }

    /// Sets the number of retry attempts for resilience.
    pub fn resilient_attempts(mut self, attempts: usize) -> Self {
        self.state.resilient_attempts = Some(attempts);
        self
    }

    /// Sets base and max backoff delays in milliseconds.
    pub fn resilient_backoff(mut self, base_delay_ms: u64, max_delay_ms: u64) -> Self {
        self.state.resilient_base_delay_ms = Some(base_delay_ms);
        self.state.resilient_max_delay_ms = Some(max_delay_ms);
        self
    }

    /// Sets jitter toggle for backoff.
    pub fn resilient_jitter(mut self, jitter: bool) -> Self {
        self.state.resilient_jitter = Some(jitter);
        self
    }
}
