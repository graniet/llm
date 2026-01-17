//! Shell tool implementation using PTY.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::tools::context::ToolContext;
use crate::tools::definition::{ToolDefinition, ToolParam};
use crate::tools::error::ToolError;
use crate::tools::pty::PtySessionManager;

/// Default shell binary.
#[cfg(unix)]
const DEFAULT_SHELL: &str = "/bin/bash";
#[cfg(windows)]
const DEFAULT_SHELL: &str = "powershell.exe";

/// Default yield time in milliseconds.
const DEFAULT_YIELD_TIME_MS: u64 = 5000;

/// Arguments for the shell tool.
#[derive(Debug, Deserialize)]
struct ShellArgs {
    /// Command to execute.
    cmd: String,
    /// Working directory (optional).
    workdir: Option<String>,
    /// Shell binary (optional).
    shell: Option<String>,
    /// Yield time in milliseconds (optional).
    yield_time_ms: Option<u64>,
}

/// Create the shell tool definition.
#[must_use]
pub fn shell_tool(pty_manager: Arc<PtySessionManager>) -> ToolDefinition {
    ToolDefinition {
        name: "shell",
        description: "Execute a shell command in a PTY. Returns output and session_id for \
                      follow-up writes. Use shell_write to send input to an existing session.",
        params: vec![
            ToolParam {
                name: "cmd",
                description: "Shell command to execute.",
                param_type: "string",
                items: None,
            },
            ToolParam {
                name: "workdir",
                description: "Working directory for the command (defaults to current dir).",
                param_type: "string",
                items: None,
            },
            ToolParam {
                name: "shell",
                description: "Shell binary to use (defaults to /bin/bash).",
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
        required: vec!["cmd"],
        executor: Arc::new(move |ctx, args| execute_shell(ctx, args, &pty_manager)),
    }
}

fn execute_shell(
    ctx: &ToolContext,
    args: Value,
    pty_manager: &PtySessionManager,
) -> Result<String, ToolError> {
    let shell_args: ShellArgs =
        serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;

    let shell = shell_args.shell.as_deref().unwrap_or(DEFAULT_SHELL);
    let workdir = shell_args.workdir.as_deref().unwrap_or(&ctx.working_dir);
    let yield_time = shell_args.yield_time_ms.unwrap_or(DEFAULT_YIELD_TIME_MS);

    // Execute in a blocking runtime context
    let result = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            pty_manager
                .spawn(shell, &shell_args.cmd, workdir, Some(yield_time))
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
