use std::fs;
use std::path::PathBuf;

use super::error::ConfigError;
use super::paths::ConfigPaths;
use super::types::AppConfig;

#[derive(Debug)]
pub struct LoadedConfig {
    pub config: AppConfig,
    pub paths: ConfigPaths,
    pub config_exists: bool,
}

pub fn load_config(path_override: Option<PathBuf>) -> Result<LoadedConfig, ConfigError> {
    let paths = ConfigPaths::resolve(path_override)?;
    ensure_dirs(&paths)?;
    let read = read_config(&paths.config_file)?;
    secure_file_permissions(&paths.config_file)?;
    Ok(LoadedConfig {
        config: read.config,
        paths,
        config_exists: read.exists,
    })
}

fn read_config(path: &PathBuf) -> Result<ConfigRead, ConfigError> {
    match fs::read_to_string(path) {
        Ok(contents) => Ok(ConfigRead {
            config: toml::from_str(&contents)?,
            exists: true,
        }),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(ConfigRead {
            config: AppConfig::default(),
            exists: false,
        }),
        Err(err) => Err(ConfigError::Io(err)),
    }
}

struct ConfigRead {
    config: AppConfig,
    exists: bool,
}

pub(super) fn ensure_dirs(paths: &ConfigPaths) -> Result<(), ConfigError> {
    fs::create_dir_all(&paths.config_dir)?;
    fs::create_dir_all(&paths.data_dir)?;
    fs::create_dir_all(&paths.logs_dir)?;
    Ok(())
}

pub(super) fn secure_file_permissions(path: &PathBuf) -> Result<(), ConfigError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = fs::metadata(path) {
            let mut perms = metadata.permissions();
            let mode = perms.mode() & 0o777;
            if mode & 0o077 != 0 {
                perms.set_mode(0o600);
                fs::set_permissions(path, perms)?;
            }
        }
    }
    Ok(())
}
