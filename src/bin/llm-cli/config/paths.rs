use std::path::PathBuf;

use super::error::ConfigError;

#[derive(Debug, Clone)]
pub struct ConfigPaths {
    pub config_file: PathBuf,
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub logs_dir: PathBuf,
}

impl ConfigPaths {
    pub fn resolve(config_override: Option<PathBuf>) -> Result<Self, ConfigError> {
        if let Some(path) = config_override {
            let dir = path
                .parent()
                .map(PathBuf::from)
                .ok_or(ConfigError::MissingHome)?;
            return Ok(Self::from_dirs(
                dir,
                default_data_dir()?,
                default_logs_dir()?,
                path,
            ));
        }
        let config_dir = default_config_dir()?;
        let config_file = config_dir.join("config.toml");
        Ok(Self::from_dirs(
            config_dir,
            default_data_dir()?,
            default_logs_dir()?,
            config_file,
        ))
    }

    fn from_dirs(
        config_dir: PathBuf,
        data_dir: PathBuf,
        logs_dir: PathBuf,
        config_file: PathBuf,
    ) -> Self {
        Self {
            config_file,
            config_dir,
            data_dir,
            logs_dir,
        }
    }

    /// Path to user-defined tools configuration
    pub fn user_tools_file(&self) -> PathBuf {
        self.config_dir.join("tools.yaml")
    }
}

fn default_config_dir() -> Result<PathBuf, ConfigError> {
    let home = dirs::home_dir().ok_or(ConfigError::MissingHome)?;
    Ok(home.join(".config").join("llm"))
}

fn default_data_dir() -> Result<PathBuf, ConfigError> {
    let home = dirs::home_dir().ok_or(ConfigError::MissingHome)?;
    Ok(home.join(".local").join("share").join("llm"))
}

fn default_logs_dir() -> Result<PathBuf, ConfigError> {
    Ok(default_data_dir()?.join("logs"))
}
