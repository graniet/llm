use serde::{Deserialize, Serialize};

use super::DEFAULT_AUTOSAVE;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct StorageConfig {
    pub autosave: bool,
    pub data_dir: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            autosave: DEFAULT_AUTOSAVE,
            data_dir: None,
        }
    }
}
