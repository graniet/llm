//! Tool execution error types.

#![allow(dead_code)]

/// Tool execution error types, inspired by Codex's FunctionCallError.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ToolError {
    /// Invalid arguments provided to the tool.
    #[error("invalid tool arguments: {0}")]
    InvalidArgs(String),

    /// Tool execution failed with a message to show the model.
    #[error("tool execution failed: {0}")]
    Execution(String),

    /// Tool not found in registry.
    #[error("tool not found: {0}")]
    NotFound(String),

    /// Tool execution was denied (sandbox/permission).
    #[error("tool denied: {0}")]
    Denied(String),

    /// Error message to send back to the model for correction.
    #[error("{0}")]
    RespondToModel(String),

    /// Fatal error that cannot be recovered.
    #[error("fatal error: {0}")]
    Fatal(String),

    /// Tool timed out.
    #[error("tool timed out after {0}ms")]
    Timeout(u64),

    /// External dependency missing (e.g., ripgrep).
    #[error("missing dependency: {0}")]
    MissingDependency(String),
}
