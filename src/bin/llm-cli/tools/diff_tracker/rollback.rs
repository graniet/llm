//! Rollback types and error handling.

#![allow(dead_code)]

use std::path::PathBuf;

/// Result of a rollback operation.
#[derive(Debug, Default)]
pub struct RollbackResult {
    /// Names of tools whose changes were rolled back.
    pub rolled_back_groups: Vec<String>,
    /// Files that were successfully restored.
    pub restored_files: Vec<PathBuf>,
    /// Errors that occurred during rollback.
    pub errors: Vec<(PathBuf, String)>,
}

impl RollbackResult {
    /// Check if the rollback was fully successful.
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }

    /// Format the result as a human-readable string.
    pub fn format(&self) -> String {
        let mut lines = Vec::new();

        if !self.rolled_back_groups.is_empty() {
            lines.push(format!(
                "Rolled back {} change group(s): {}",
                self.rolled_back_groups.len(),
                self.rolled_back_groups.join(", ")
            ));
        }

        if !self.restored_files.is_empty() {
            lines.push(format!("Restored {} file(s):", self.restored_files.len()));
            for path in &self.restored_files {
                lines.push(format!("  - {}", path.display()));
            }
        }

        if !self.errors.is_empty() {
            lines.push(format!("Errors ({}):", self.errors.len()));
            for (path, error) in &self.errors {
                lines.push(format!("  - {}: {}", path.display(), error));
            }
        }

        lines.join("\n")
    }
}

/// Error during rollback.
#[derive(Debug)]
pub enum RollbackError {
    /// No changes to rollback.
    NoChanges,
    /// IO error during rollback.
    Io(std::io::Error),
}

impl std::fmt::Display for RollbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoChanges => write!(f, "No changes to rollback"),
            Self::Io(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for RollbackError {}
