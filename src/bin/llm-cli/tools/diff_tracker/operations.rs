//! File system operations for rollback.

use std::fs;

use super::types::{ChangeType, FileChange};

/// Rollback a single file change.
pub fn rollback_change(change: &FileChange) -> Result<(), String> {
    match &change.change_type {
        ChangeType::Created => {
            fs::remove_file(&change.path).map_err(|e| format!("Failed to delete file: {e}"))?;
        }
        ChangeType::Modified => {
            let content = change
                .original_content
                .as_ref()
                .ok_or("No original content recorded")?;
            fs::write(&change.path, content).map_err(|e| format!("Failed to restore file: {e}"))?;
        }
        ChangeType::Deleted => {
            let content = change
                .original_content
                .as_ref()
                .ok_or("No original content recorded")?;
            if let Some(parent) = change.path.parent() {
                fs::create_dir_all(parent).map_err(|e| format!("Failed to create dir: {e}"))?;
            }
            fs::write(&change.path, content).map_err(|e| format!("Failed to restore file: {e}"))?;
        }
        ChangeType::Renamed { from } => {
            let content = change
                .original_content
                .as_ref()
                .ok_or("No original content recorded")?;
            if change.path.exists() {
                fs::remove_file(&change.path).map_err(|e| format!("Failed to delete: {e}"))?;
            }
            if let Some(parent) = from.parent() {
                fs::create_dir_all(parent).map_err(|e| format!("Failed to create dir: {e}"))?;
            }
            fs::write(from, content).map_err(|e| format!("Failed to restore file: {e}"))?;
        }
    }
    Ok(())
}
