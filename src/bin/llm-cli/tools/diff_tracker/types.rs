//! Types for diff tracking.

#![allow(dead_code)]

use std::path::PathBuf;

/// Type of file change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeType {
    /// File was created.
    Created,
    /// File was modified.
    Modified,
    /// File was deleted.
    Deleted,
    /// File was renamed/moved.
    Renamed { from: PathBuf },
}

/// A tracked file change.
#[derive(Debug, Clone)]
pub struct FileChange {
    /// Path to the file.
    pub path: PathBuf,
    /// Type of change.
    pub change_type: ChangeType,
    /// Original content (None if file was created).
    pub original_content: Option<String>,
    /// Timestamp of the change.
    pub timestamp: std::time::SystemTime,
}

/// A change group representing a single tool execution.
#[derive(Debug, Clone)]
pub struct ChangeGroup {
    /// Tool name that made the changes.
    pub tool_name: String,
    /// Description of the operation.
    pub description: String,
    /// Changes in this group.
    pub changes: Vec<FileChange>,
    /// Timestamp when the group was created.
    pub timestamp: std::time::SystemTime,
}

impl ChangeGroup {
    /// Create a new change group.
    pub fn new(tool_name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            description: description.into(),
            changes: Vec::new(),
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Add a file change to this group.
    pub fn add_change(&mut self, change: FileChange) {
        self.changes.push(change);
    }
}

/// Summary of a change group.
#[derive(Debug, Clone)]
pub struct ChangeSummary {
    /// Index in the change list.
    pub index: usize,
    /// Tool that made the changes.
    pub tool_name: String,
    /// Description of the operation.
    pub description: String,
    /// Number of files affected.
    pub file_count: usize,
    /// Timestamp.
    pub timestamp: std::time::SystemTime,
}
