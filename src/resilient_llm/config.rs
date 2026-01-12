/// Configuration for retry and backoff behavior.
#[derive(Clone, Debug)]
pub struct ResilienceConfig {
    /// Maximum number of attempts including the first one
    pub max_attempts: usize,
    /// Initial backoff delay in milliseconds
    pub base_delay_ms: u64,
    /// Maximum backoff delay in milliseconds
    pub max_delay_ms: u64,
    /// Whether to add random jitter to backoff delays
    pub jitter: bool,
}

const DEFAULT_MAX_ATTEMPTS: usize = 3;
const DEFAULT_BASE_DELAY_MS: u64 = 200;
const DEFAULT_MAX_DELAY_MS: u64 = 2_000;

impl ResilienceConfig {
    /// Creates a default configuration with sane values.
    pub fn defaults() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            base_delay_ms: DEFAULT_BASE_DELAY_MS,
            max_delay_ms: DEFAULT_MAX_DELAY_MS,
            jitter: true,
        }
    }
}
