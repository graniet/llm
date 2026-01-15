use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::{ChatConfig, LoggingConfig, ProviderConfig, StorageConfig, ToolsConfig, UiConfig};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct AppConfig {
    pub default_provider: Option<String>,
    pub default_model: Option<String>,
    pub providers: BTreeMap<String, ProviderConfig>,
    pub chat: ChatConfig,
    pub ui: UiConfig,
    pub tools: ToolsConfig,
    pub storage: StorageConfig,
    pub logging: LoggingConfig,
}
