//! Diff tracking for file changes with rollback support.
//!
//! Tracks file modifications made by tools so they can be rolled back.

#![allow(dead_code)]

mod operations;
mod rollback;
mod types;

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::RwLock;

use operations::rollback_change;
pub use rollback::{RollbackError, RollbackResult};
pub use types::{ChangeGroup, ChangeSummary, ChangeType, FileChange};

/// Diff tracker for managing file change history.
#[derive(Debug)]
pub struct DiffTracker {
    groups: RwLock<Vec<ChangeGroup>>,
    max_groups: usize,
}

impl Default for DiffTracker {
    fn default() -> Self {
        Self::new(100)
    }
}

impl DiffTracker {
    /// Create a new diff tracker with the specified maximum group count.
    pub fn new(max_groups: usize) -> Self {
        Self {
            groups: RwLock::new(Vec::new()),
            max_groups,
        }
    }

    /// Record a file being created.
    pub async fn record_create(
        &self,
        path: impl Into<PathBuf>,
        tool_name: impl Into<String>,
        description: impl Into<String>,
    ) {
        let change = FileChange {
            path: path.into(),
            change_type: ChangeType::Created,
            original_content: None,
            timestamp: std::time::SystemTime::now(),
        };
        let mut group = ChangeGroup::new(tool_name, description);
        group.add_change(change);
        self.add_group(group).await;
    }

    /// Record a file being modified.
    pub async fn record_modify(
        &self,
        path: impl Into<PathBuf>,
        original_content: impl Into<String>,
        tool_name: impl Into<String>,
        description: impl Into<String>,
    ) {
        let change = FileChange {
            path: path.into(),
            change_type: ChangeType::Modified,
            original_content: Some(original_content.into()),
            timestamp: std::time::SystemTime::now(),
        };
        let mut group = ChangeGroup::new(tool_name, description);
        group.add_change(change);
        self.add_group(group).await;
    }

    /// Record a file being deleted.
    pub async fn record_delete(
        &self,
        path: impl Into<PathBuf>,
        original_content: impl Into<String>,
        tool_name: impl Into<String>,
        description: impl Into<String>,
    ) {
        let change = FileChange {
            path: path.into(),
            change_type: ChangeType::Deleted,
            original_content: Some(original_content.into()),
            timestamp: std::time::SystemTime::now(),
        };
        let mut group = ChangeGroup::new(tool_name, description);
        group.add_change(change);
        self.add_group(group).await;
    }

    /// Add a change group directly.
    pub async fn add_group(&self, group: ChangeGroup) {
        let mut groups = self.groups.write().await;
        groups.push(group);
        while groups.len() > self.max_groups {
            groups.remove(0);
        }
    }

    /// Rollback the last N change groups.
    pub async fn rollback(&self, count: usize) -> Result<RollbackResult, RollbackError> {
        let mut groups = self.groups.write().await;
        if groups.is_empty() {
            return Err(RollbackError::NoChanges);
        }
        let count = count.min(groups.len());
        let mut result = RollbackResult::default();

        for _ in 0..count {
            if let Some(group) = groups.pop() {
                for change in group.changes.into_iter().rev() {
                    match rollback_change(&change) {
                        Ok(()) => result.restored_files.push(change.path),
                        Err(e) => result.errors.push((change.path, e)),
                    }
                }
                result.rolled_back_groups.push(group.tool_name);
            }
        }
        Ok(result)
    }

    /// Get a summary of pending changes.
    pub async fn summary(&self) -> Vec<ChangeSummary> {
        let groups = self.groups.read().await;
        groups
            .iter()
            .enumerate()
            .map(|(idx, group)| ChangeSummary {
                index: idx,
                tool_name: group.tool_name.clone(),
                description: group.description.clone(),
                file_count: group.changes.len(),
                timestamp: group.timestamp,
            })
            .collect()
    }

    /// Get the number of change groups.
    pub async fn group_count(&self) -> usize {
        self.groups.read().await.len()
    }

    /// Clear all tracked changes.
    pub async fn clear(&self) {
        self.groups.write().await.clear();
    }
}
/// Create a shared diff tracker.
pub fn create_tracker() -> Arc<DiffTracker> {
    Arc::new(DiffTracker::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_record_create_and_rollback() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "new content").unwrap();

        let tracker = DiffTracker::new(10);
        tracker
            .record_create(&file_path, "patch", "create test.txt")
            .await;
        assert!(file_path.exists());

        let result = tracker.rollback(1).await.unwrap();
        assert!(result.is_success());
        assert!(!file_path.exists());
    }

    #[tokio::test]
    async fn test_record_modify_and_rollback() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "original").unwrap();

        let tracker = DiffTracker::new(10);
        tracker
            .record_modify(&file_path, "original", "patch", "modify test.txt")
            .await;
        fs::write(&file_path, "modified").unwrap();

        let result = tracker.rollback(1).await.unwrap();
        assert!(result.is_success());
        assert_eq!(fs::read_to_string(&file_path).unwrap(), "original");
    }
}
