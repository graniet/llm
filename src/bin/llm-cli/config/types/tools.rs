use serde::{Deserialize, Serialize};

use super::DEFAULT_TOOL_TIMEOUT_MS;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ToolsConfig {
    pub execution: ToolExecutionMode,
    pub enabled: Vec<String>,
    pub allowed_paths: Vec<String>,
    pub timeout_ms: u64,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            execution: ToolExecutionMode::Ask,
            enabled: Vec::new(),
            allowed_paths: Vec::new(),
            timeout_ms: DEFAULT_TOOL_TIMEOUT_MS,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionMode {
    Always,
    Ask,
    Never,
}
