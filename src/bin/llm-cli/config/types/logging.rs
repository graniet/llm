use serde::{Deserialize, Serialize};

use super::{DEFAULT_LOG_ROTATE_KEEP, DEFAULT_LOG_ROTATE_SIZE};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: String,
    pub path: Option<String>,
    pub rotate_size: u64,
    pub rotate_keep: usize,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            path: None,
            rotate_size: DEFAULT_LOG_ROTATE_SIZE,
            rotate_keep: DEFAULT_LOG_ROTATE_KEEP,
        }
    }
}
