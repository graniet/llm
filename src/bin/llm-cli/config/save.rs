use std::fs;

use super::error::ConfigError;
use super::load::{ensure_dirs, secure_file_permissions};
use super::paths::ConfigPaths;
use super::types::AppConfig;

pub fn save_config(config: &AppConfig, paths: &ConfigPaths) -> Result<(), ConfigError> {
    ensure_dirs(paths)?;
    let contents = toml::to_string_pretty(config)?;
    fs::write(&paths.config_file, contents)?;
    secure_file_permissions(&paths.config_file)?;
    Ok(())
}
