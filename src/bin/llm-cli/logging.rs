use std::path::PathBuf;

use flexi_logger::{Cleanup, Criterion, FileSpec, Logger, Naming};

use crate::config::{ConfigPaths, LoggingConfig};

pub fn init_logging(config: &LoggingConfig, paths: &ConfigPaths) -> anyhow::Result<()> {
    let log_path = config
        .path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| paths.logs_dir.join("llm.log"));
    let directory = log_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or(paths.logs_dir.clone());
    let basename = log_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("llm")
        .to_string();
    Logger::try_with_env_or_str(&config.level)?
        .log_to_file(FileSpec::default().directory(directory).basename(basename))
        .rotate(
            Criterion::Size(config.rotate_size),
            Naming::Numbers,
            Cleanup::KeepLogFiles(config.rotate_keep),
        )
        .start()?;
    Ok(())
}
