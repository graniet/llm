use std::sync::Arc;

use chrono::Utc;
use serde_json::Value;

use super::context::ToolContext;
use super::definition::{ToolDefinition, ToolParam};
use super::diff_tracker::DiffTracker;
use super::error::ToolError;
use super::handlers::{
    file_read_tool, ls_tool, patch_tool, plan_tool, rollback_tool, search_tool, shell_tool,
    shell_write_tool,
};
use super::pty::PtySessionManager;

/// Create all built-in tools.
///
/// The `pty_manager` is optional - if None, shell tools are not included.
pub fn builtin_tools() -> Vec<ToolDefinition> {
    vec![echo_tool(), time_tool()]
}

/// Create all built-in tools including shell tools with PTY support.
pub fn builtin_tools_with_pty(
    pty_manager: Arc<PtySessionManager>,
    diff_tracker: Arc<DiffTracker>,
) -> Vec<ToolDefinition> {
    vec![
        // Legacy tools
        echo_tool(),
        time_tool(),
        // New Codex-style tools
        shell_tool(Arc::clone(&pty_manager)),
        shell_write_tool(pty_manager),
        file_read_tool(),
        search_tool(),
        ls_tool(),
        patch_tool(),
        plan_tool(),
        rollback_tool(diff_tracker),
    ]
}

fn echo_tool() -> ToolDefinition {
    ToolDefinition {
        name: "echo",
        description: "Echo back the provided text.",
        params: vec![ToolParam {
            name: "text",
            description: "Text to echo back.",
            param_type: "string",
            items: None,
        }],
        required: vec!["text"],
        executor: Arc::new(exec_echo),
    }
}

fn time_tool() -> ToolDefinition {
    ToolDefinition {
        name: "time_now",
        description: "Return the current UTC time in RFC3339 format.",
        params: Vec::new(),
        required: Vec::new(),
        executor: Arc::new(exec_time),
    }
}

fn exec_echo(_ctx: &ToolContext, args: Value) -> Result<String, ToolError> {
    let text = args
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::InvalidArgs("missing 'text'".to_string()))?;
    Ok(text.to_string())
}

fn exec_time(_ctx: &ToolContext, _args: Value) -> Result<String, ToolError> {
    Ok(Utc::now().to_rfc3339())
}
