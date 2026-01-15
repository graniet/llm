use std::future::Future;
use std::time::Duration;

use tokio::time::sleep;

use crate::{error::LLMError, LLMProvider};

use super::config::ResilienceConfig;

/// Resilient wrapper that retries transient failures using exponential backoff.
pub struct ResilientLLM {
    pub(super) inner: Box<dyn LLMProvider>,
    pub(super) cfg: ResilienceConfig,
}

impl ResilientLLM {
    /// Creates a new resilient wrapper around an existing provider.
    pub fn new(inner: Box<dyn LLMProvider>, cfg: ResilienceConfig) -> Self {
        Self { inner, cfg }
    }

    pub(super) async fn retry<F, Fut, T>(&self, mut op: F) -> Result<T, LLMError>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, LLMError>>,
    {
        let mut attempts_left = self.cfg.max_attempts;
        let mut idx = 0usize;
        let mut last_err: Option<LLMError> = None;

        while attempts_left > 0 {
            match op().await {
                Ok(value) => return Ok(value),
                Err(err) => {
                    if attempts_left == 1 || !Self::is_retryable(&err) {
                        return Err(err);
                    }
                    last_err = Some(err);
                    self.backoff_sleep(idx).await;
                    attempts_left -= 1;
                    idx += 1;
                }
            }
        }

        Err(LLMError::RetryExceeded {
            attempts: self.cfg.max_attempts,
            last_error: last_err.map(|e| e.to_string()).unwrap_or_default(),
        })
    }

    fn is_retryable(err: &LLMError) -> bool {
        match err {
            LLMError::HttpError(_) => true,
            LLMError::ProviderError(_) => true,
            LLMError::ResponseFormatError { .. } => true,
            LLMError::JsonError(_) => true,
            LLMError::Generic(_) => true,
            LLMError::RetryExceeded { .. } => false,
            LLMError::AuthError(_) => false,
            LLMError::InvalidRequest(_) => false,
            LLMError::ToolConfigError(_) => false,
        }
    }

    async fn backoff_sleep(&self, attempt_index: usize) {
        let mut delay = self
            .cfg
            .base_delay_ms
            .saturating_mul(1u64 << attempt_index.min(16));
        delay = delay.min(self.cfg.max_delay_ms);
        if self.cfg.jitter {
            let span = (delay / 2).max(1);
            let jitter = ((attempt_index as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1))
                % span;
            delay = delay.saturating_sub(jitter);
        }
        sleep(Duration::from_millis(delay)).await;
    }
}

impl LLMProvider for ResilientLLM {}
