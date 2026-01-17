//! Command execution for user-defined tools.

use serde_json::Value;

use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;

/// Execute a command-based tool with parameter substitution.
pub fn execute_command_tool(
    command_template: &str,
    tool_name: &str,
    ctx: &ToolContext,
    args: Value,
) -> Result<String, ToolError> {
    // Substitute parameters in command template
    let mut command = command_template.to_string();

    if let Value::Object(map) = &args {
        for (key, value) in map {
            let placeholder = format!("{{{{{}}}}}", key);
            let replacement = match value {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                _ => value.to_string(),
            };
            command = command.replace(&placeholder, &replacement);
        }
    }

    // Check for missing parameters
    if command.contains("{{") && command.contains("}}") {
        return Err(ToolError::InvalidArgs(
            "Missing required parameters in command".to_string(),
        ));
    }

    // Execute command
    // Use bash if available for better compatibility, fallback to sh
    let shell = if std::path::Path::new("/bin/bash").exists() {
        "/bin/bash"
    } else {
        "/bin/sh"
    };

    let output = std::process::Command::new(shell)
        .arg("-c")
        .arg(&command)
        .current_dir(&ctx.working_dir)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| ToolError::Execution(format!("Failed to execute {}: {}", tool_name, e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if output.status.success() {
        // Return stdout, or stderr if stdout is empty (some commands output to stderr)
        if stdout.is_empty() && !stderr.is_empty() {
            Ok(stderr.to_string())
        } else {
            Ok(stdout.to_string())
        }
    } else {
        // Include both stdout and stderr in error for debugging
        let mut error_msg = format!("{} exited with code {:?}", tool_name, output.status.code());
        if !stderr.is_empty() {
            error_msg.push_str(&format!("\nstderr: {}", stderr));
        }
        if !stdout.is_empty() {
            error_msg.push_str(&format!("\nstdout: {}", stdout));
        }
        Err(ToolError::Execution(error_msg))
    }
}
