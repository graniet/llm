//! Directory listing tool with recursive traversal.

#![allow(dead_code)]

mod traversal;
mod types;

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use crate::tools::context::ToolContext;
use crate::tools::definition::{ToolDefinition, ToolParam};
use crate::tools::error::ToolError;

use traversal::{collect_entries, slice_entries};
use types::LsArgs;

/// Create the ls tool definition.
#[must_use]
pub fn ls_tool() -> ToolDefinition {
    ToolDefinition {
        name: "ls",
        description: "List directory contents with recursive traversal. \
                      Returns file names with type indicators (/ for dirs, @ for symlinks).",
        params: vec![
            ToolParam {
                name: "dir_path",
                description: "Absolute path to the directory to list.",
                param_type: "string",
                items: None,
            },
            ToolParam {
                name: "offset",
                description: "1-indexed entry number to start from (default: 1).",
                param_type: "number",
                items: None,
            },
            ToolParam {
                name: "limit",
                description: "Maximum entries to return (default: 25).",
                param_type: "number",
                items: None,
            },
            ToolParam {
                name: "depth",
                description: "Maximum directory depth (default: 2).",
                param_type: "number",
                items: None,
            },
        ],
        required: vec!["dir_path"],
        executor: Arc::new(execute_ls),
    }
}

fn execute_ls(_ctx: &ToolContext, args: Value) -> Result<String, ToolError> {
    let ls_args: LsArgs =
        serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;

    if ls_args.offset == 0 {
        return Err(ToolError::RespondToModel(
            "offset must be a 1-indexed entry number".to_string(),
        ));
    }

    if ls_args.limit == 0 {
        return Err(ToolError::RespondToModel(
            "limit must be greater than zero".to_string(),
        ));
    }

    if ls_args.depth == 0 {
        return Err(ToolError::RespondToModel(
            "depth must be greater than zero".to_string(),
        ));
    }

    let path = PathBuf::from(&ls_args.dir_path);
    if !path.is_absolute() {
        return Err(ToolError::RespondToModel(
            "dir_path must be an absolute path".to_string(),
        ));
    }

    if !path.is_dir() {
        return Err(ToolError::RespondToModel(
            "dir_path is not a directory".to_string(),
        ));
    }

    let entries = collect_entries(&path, ls_args.depth)?;
    let slice = slice_entries(entries, ls_args.offset, ls_args.limit)?;

    let mut output = vec![format!("Absolute path: {}", path.display())];
    output.extend(slice);
    Ok(output.join("\n"))
}
