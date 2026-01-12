use std::path::PathBuf;

use crate::diff::{apply_diff, DiffView};
use crate::runtime::AppStatus;

use super::AppController;

impl AppController {
    pub fn apply_diff(&mut self, diff: DiffView) -> bool {
        let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        match apply_diff(&diff, &base) {
            Ok(paths) => {
                let summary = if paths.is_empty() {
                    "Diff applied.".to_string()
                } else {
                    format!("Diff applied to: {}", format_paths(&paths))
                };
                self.push_notice(summary);
                true
            }
            Err(err) => {
                self.set_status(AppStatus::Error(format!("apply diff: {err}")));
                false
            }
        }
    }
}

fn format_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}
