//! Tool system for LLM-CLI.
//!
//! This module provides the tool infrastructure including handlers,
//! sandbox permissions, PTY sessions, and diff tracking.

// Public API exports - may not be used internally but exposed for consumers
#![allow(unused_imports)]

mod builtin;
mod context;
mod definition;
pub mod diff_tracker;
mod error;
pub mod handlers;
pub mod parallel;
pub mod pty;
mod registry;
pub mod sandbox;
mod user_tools;

pub use context::ToolContext;
pub use definition::{ToolDefinition, ToolExecutor, ToolParam};
pub use diff_tracker::{create_tracker, DiffTracker};
pub use error::ToolError;
pub use handlers::{
    file_read_tool, ls_tool, patch_tool, plan_tool, rollback_tool, search_tool, shell_tool,
    shell_write_tool,
};
pub use parallel::{is_mutating_tool, ParallelConfig, ParallelExecutor};
pub use pty::PtySessionManager;
pub use registry::ToolRegistry;
pub use sandbox::{is_safe_command, SandboxLevel, SandboxPermissions};
pub use user_tools::{UserTool, UserToolsConfig};
