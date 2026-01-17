//! File read tool with indentation-aware mode.

#![allow(dead_code)]

mod indentation;
mod slice;
mod types;

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;

use crate::tools::context::ToolContext;
use crate::tools::definition::{ToolDefinition, ToolParam};
use crate::tools::error::ToolError;

use indentation::read_indentation;
use slice::read_slice;
use types::{FileReadArgs, ReadMode};

/// Create the file_read tool definition.
#[must_use]
pub fn file_read_tool() -> ToolDefinition {
    ToolDefinition {
        name: "file_read",
        description: "Read a file with optional indentation-aware block extraction. \
                      Use mode='indentation' to extract code blocks based on indentation.",
        params: vec![
            ToolParam {
                name: "file_path",
                description: "Absolute path to the file to read.",
                param_type: "string",
                items: None,
            },
            ToolParam {
                name: "offset",
                description: "1-indexed line number to start from (default: 1).",
                param_type: "number",
                items: None,
            },
            ToolParam {
                name: "limit",
                description: "Maximum lines to return (default: 2000).",
                param_type: "number",
                items: None,
            },
            ToolParam {
                name: "mode",
                description: "Read mode: 'slice' (default) or 'indentation'.",
                param_type: "string",
                items: None,
            },
        ],
        required: vec!["file_path"],
        executor: Arc::new(execute_file_read),
    }
}

fn execute_file_read(_ctx: &ToolContext, args: Value) -> Result<String, ToolError> {
    let read_args: FileReadArgs =
        serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;

    if read_args.offset == 0 {
        return Err(ToolError::RespondToModel(
            "offset must be a 1-indexed line number".to_string(),
        ));
    }

    if read_args.limit == 0 {
        return Err(ToolError::RespondToModel(
            "limit must be greater than zero".to_string(),
        ));
    }

    let path = Path::new(&read_args.file_path);
    if !path.is_absolute() {
        return Err(ToolError::RespondToModel(
            "file_path must be an absolute path".to_string(),
        ));
    }

    let result = match read_args.mode {
        ReadMode::Slice => read_slice(path, read_args.offset, read_args.limit),
        ReadMode::Indentation => {
            let opts = read_args.indentation.unwrap_or_default();
            read_indentation(path, read_args.offset, read_args.limit, opts)
        }
    }?;

    Ok(result.join("\n"))
}
