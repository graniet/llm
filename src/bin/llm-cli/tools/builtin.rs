use std::sync::Arc;

use chrono::Utc;
use serde_json::Value;

use super::context::ToolContext;
use super::definition::{ToolDefinition, ToolParam};
use super::error::ToolError;

pub fn builtin_tools() -> Vec<ToolDefinition> {
    vec![echo_tool(), time_tool()]
}

fn echo_tool() -> ToolDefinition {
    ToolDefinition {
        name: "echo",
        description: "Echo back the provided text.",
        params: vec![ToolParam {
            name: "text",
            description: "Text to echo back.",
            param_type: "string",
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
