//! Sandbox permission levels for tool execution.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Sandbox permission level for tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxLevel {
    /// Read-only access - no file modifications allowed.
    ReadOnly,
    /// Workspace write access - can modify files in workspace only.
    #[default]
    WorkspaceWrite,
    /// Full access - dangerous, allows all operations.
    FullAccess,
}

impl SandboxLevel {
    /// Check if this level allows write operations.
    #[must_use]
    pub const fn allows_write(&self) -> bool {
        matches!(self, Self::WorkspaceWrite | Self::FullAccess)
    }

    /// Check if this level allows operations outside workspace.
    #[must_use]
    pub const fn allows_outside_workspace(&self) -> bool {
        matches!(self, Self::FullAccess)
    }
}

/// Sandbox permissions context for a tool execution.
#[derive(Debug, Clone)]
pub struct SandboxPermissions {
    /// Current sandbox level.
    pub level: SandboxLevel,
    /// Workspace root path for workspace-write level.
    pub workspace_root: Option<String>,
}

impl Default for SandboxPermissions {
    fn default() -> Self {
        Self {
            level: SandboxLevel::WorkspaceWrite,
            workspace_root: None,
        }
    }
}

impl SandboxPermissions {
    /// Create new sandbox permissions with the given level.
    #[must_use]
    pub const fn new(level: SandboxLevel) -> Self {
        Self {
            level,
            workspace_root: None,
        }
    }

    /// Set the workspace root path.
    #[must_use]
    pub fn with_workspace(mut self, root: String) -> Self {
        self.workspace_root = Some(root);
        self
    }

    /// Check if a path is allowed for write operations.
    #[must_use]
    pub fn is_write_allowed(&self, path: &str) -> bool {
        match self.level {
            SandboxLevel::ReadOnly => false,
            SandboxLevel::FullAccess => true,
            SandboxLevel::WorkspaceWrite => self
                .workspace_root
                .as_ref()
                .is_none_or(|root| path.starts_with(root)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_denies_writes() {
        let perms = SandboxPermissions::new(SandboxLevel::ReadOnly);
        assert!(!perms.is_write_allowed("/any/path"));
    }

    #[test]
    fn full_access_allows_all() {
        let perms = SandboxPermissions::new(SandboxLevel::FullAccess);
        assert!(perms.is_write_allowed("/any/path"));
    }

    #[test]
    fn workspace_write_respects_root() {
        let perms = SandboxPermissions::new(SandboxLevel::WorkspaceWrite)
            .with_workspace("/home/user/project".to_string());

        assert!(perms.is_write_allowed("/home/user/project/src/main.rs"));
        assert!(!perms.is_write_allowed("/etc/passwd"));
    }
}
