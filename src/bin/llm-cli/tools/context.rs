//! Tool execution context.

#![allow(dead_code)]

use super::sandbox::{SandboxLevel, SandboxPermissions};

/// Context provided to tool executors.
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    /// Allowed paths for file operations.
    pub allowed_paths: Vec<String>,
    /// Default timeout in milliseconds.
    pub timeout_ms: u64,
    /// Current working directory.
    pub working_dir: String,
    /// Sandbox permissions for this execution.
    pub sandbox: SandboxPermissions,
}

impl ToolContext {
    /// Create a new tool context with working directory.
    #[must_use]
    pub fn new(working_dir: String) -> Self {
        Self {
            working_dir: working_dir.clone(),
            sandbox: SandboxPermissions::default().with_workspace(working_dir),
            ..Default::default()
        }
    }

    /// Set the sandbox level.
    #[must_use]
    pub fn with_sandbox_level(mut self, level: SandboxLevel) -> Self {
        self.sandbox.level = level;
        self
    }

    /// Set the timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set allowed paths.
    #[must_use]
    pub fn with_allowed_paths(mut self, paths: Vec<String>) -> Self {
        self.allowed_paths = paths;
        self
    }

    /// Check if a path is allowed for write operations.
    #[must_use]
    pub fn is_write_allowed(&self, path: &str) -> bool {
        self.sandbox.is_write_allowed(path)
    }
}
