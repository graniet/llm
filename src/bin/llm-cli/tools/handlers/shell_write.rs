//! Shell write tool for sending input to an existing PTY session.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::tools::context::ToolContext;
use crate::tools::definition::{ToolDefinition, ToolParam};
use crate::tools::error::ToolError;
use crate::tools::pty::{PtySessionManager, SessionId};

/// Default yield time in milliseconds.
const DEFAULT_YIELD_TIME_MS: u64 = 5000;

/// Arguments for the shell_write tool.
#[derive(Debug, Deserialize)]
struct ShellWriteArgs {
    /// Session ID from a previous shell call.
    session_id: u64,
    /// Characters to send to the session.
    chars: String,
    /// Yield time in milliseconds (optional).
    yield_time_ms: Option<u64>,
}

/// Create the shell_write tool definition.
#[must_use]
pub fn shell_write_tool(pty_manager: Arc<PtySessionManager>) -> ToolDefinition {
    ToolDefinition {
        name: "shell_write",
        description: "Write characters to an existing PTY session. Use the session_id from a \
                      previous shell call. Returns new output from the session.",
        params: vec![
            ToolParam {
                name: "session_id",
                description: "Session ID from a previous shell call.",
                param_type: "number",
                items: None,
            },
            ToolParam {
                name: "chars",
                description: "Characters to send to the session (can include \\n for enter).",
                param_type: "string",
                items: None,
            },
            ToolParam {
                name: "yield_time_ms",
                description: "How long to wait for output before returning (default 5000ms).",
                param_type: "number",
                items: None,
            },
        ],
        required: vec!["session_id", "chars"],
        executor: Arc::new(move |ctx, args| execute_shell_write(ctx, args, &pty_manager)),
    }
}

fn execute_shell_write(
    _ctx: &ToolContext,
    args: Value,
    pty_manager: &PtySessionManager,
) -> Result<String, ToolError> {
    let write_args: ShellWriteArgs =
        serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;

    let session_id = SessionId(write_args.session_id);
    let yield_time = write_args.yield_time_ms.unwrap_or(DEFAULT_YIELD_TIME_MS);

    // Execute in a blocking runtime context
    let result = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            pty_manager
                .write(session_id, &write_args.chars, Some(yield_time))
                .await
        })
    })?;

    // Format output as JSON
    let output = json!({
        "output": result.output,
        "metadata": {
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_secs,
            "session_id": result.session_id.0,
            "has_exited": result.has_exited
        }
    });

    Ok(output.to_string())
}
