//! Search tool using ripgrep.

use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::tools::context::ToolContext;
use crate::tools::definition::{ToolDefinition, ToolParam};
use crate::tools::error::ToolError;

/// Default limit for results.
const DEFAULT_LIMIT: usize = 100;

/// Maximum limit.
const MAX_LIMIT: usize = 2000;

/// Search timeout in seconds.
const SEARCH_TIMEOUT_SECS: u64 = 30;

/// Arguments for the search tool.
#[derive(Debug, Deserialize)]
struct SearchArgs {
    /// Regex pattern to search for.
    pattern: String,
    /// Glob pattern to filter files (e.g., "*.rs").
    include: Option<String>,
    /// Directory to search in.
    path: Option<String>,
    /// Maximum number of file paths to return.
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    DEFAULT_LIMIT
}

/// Create the search tool definition.
#[must_use]
pub fn search_tool() -> ToolDefinition {
    ToolDefinition {
        name: "search",
        description: "Search for files matching a regex pattern using ripgrep. \
                      Returns file paths containing matches. Requires ripgrep (rg) installed.",
        params: vec![
            ToolParam {
                name: "pattern",
                description: "Regex pattern to search for.",
                param_type: "string",
                items: None,
            },
            ToolParam {
                name: "include",
                description: "Glob pattern to filter files (e.g., '*.rs', '*.py').",
                param_type: "string",
                items: None,
            },
            ToolParam {
                name: "path",
                description: "Directory to search in (defaults to working directory).",
                param_type: "string",
                items: None,
            },
            ToolParam {
                name: "limit",
                description: "Maximum file paths to return (default: 100, max: 2000).",
                param_type: "number",
                items: None,
            },
        ],
        required: vec!["pattern"],
        executor: Arc::new(execute_search),
    }
}

fn execute_search(ctx: &ToolContext, args: Value) -> Result<String, ToolError> {
    let search_args: SearchArgs =
        serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;

    // Check ripgrep availability
    check_ripgrep_available()?;

    let search_path = search_args.path.as_deref().unwrap_or(&ctx.working_dir);
    let limit = search_args.limit.min(MAX_LIMIT);

    // Build ripgrep command
    let mut cmd = Command::new("rg");
    cmd.arg("--files-with-matches")
        .arg("--no-messages")
        .arg("--color=never");

    if let Some(ref glob) = search_args.include {
        cmd.arg("--glob").arg(glob);
    }

    cmd.arg(&search_args.pattern).arg(search_path);

    // Execute with timeout
    let output = execute_with_timeout(&mut cmd, Duration::from_secs(SEARCH_TIMEOUT_SECS))?;

    // Parse output
    let stdout = String::from_utf8_lossy(&output.stdout);
    let files: Vec<&str> = stdout.lines().take(limit).collect();

    if files.is_empty() {
        return Ok("No matches found.".to_string());
    }

    // Sort by modification time (most recent first) - best effort
    let mut result = format!("Found {} file(s) matching pattern:\n", files.len());
    for file in &files {
        result.push_str(file);
        result.push('\n');
    }

    if stdout.lines().count() > limit {
        result.push_str(&format!("\n(Results truncated to {} files)", limit));
    }

    Ok(result)
}

fn check_ripgrep_available() -> Result<(), ToolError> {
    match Command::new("rg").arg("--version").output() {
        Ok(output) if output.status.success() => Ok(()),
        _ => Err(ToolError::MissingDependency(
            "ripgrep (rg) is not installed. Install with: brew install ripgrep (macOS), \
             apt install ripgrep (Ubuntu), or cargo install ripgrep"
                .to_string(),
        )),
    }
}

fn execute_with_timeout(
    cmd: &mut Command,
    timeout: Duration,
) -> Result<std::process::Output, ToolError> {
    use std::process::Stdio;

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| ToolError::Execution(format!("Failed to spawn rg: {e}")))?;

    // Wait with timeout
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    return Err(ToolError::Timeout(timeout.as_millis() as u64));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(ToolError::Execution(format!("Wait failed: {e}"))),
        }
    }

    child
        .wait_with_output()
        .map_err(|e| ToolError::Execution(format!("Failed to get output: {e}")))
}
