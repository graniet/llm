mod app;
mod chat;
mod logging;
mod provider;
mod storage;
mod tools;
mod ui;

const DEFAULT_AUTOSAVE: bool = true;
const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 8_000;
const DEFAULT_TOOL_TIMEOUT_MS: u64 = 5_000;
const DEFAULT_LOG_ROTATE_SIZE: u64 = 10 * 1024 * 1024;
const DEFAULT_LOG_ROTATE_KEEP: usize = 5;
const DEFAULT_AUTO_COMPACT_THRESHOLD: f32 = 0.9;

pub use app::AppConfig;
pub use chat::{ChatConfig, TrimStrategy};
pub use logging::LoggingConfig;
pub use provider::{ModelConfig, PricingConfig, ProviderConfig};
pub use storage::StorageConfig;
pub use tools::{ToolExecutionMode, ToolsConfig};
pub use ui::{NavigationMode, UiConfig};
