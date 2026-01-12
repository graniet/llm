use std::io;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config IO error: {0}")]
    Io(#[from] io::Error),
    #[error("config parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("config serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
    #[error("missing home directory for config paths")]
    MissingHome,
}
