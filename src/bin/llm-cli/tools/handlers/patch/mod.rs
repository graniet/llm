//! Patch tool for applying file modifications.
//!
//! Supports two formats:
//! - JSON: structured patch format
//! - Freeform: Codex-style text format with pest parser

mod executor;
mod modification;
mod parser;
mod types;

use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use crate::tools::context::ToolContext;
use crate::tools::definition::{ToolDefinition, ToolParam};
use crate::tools::error::ToolError;

use executor::apply_patch;
use parser::parse_freeform_patch;
use types::{Patch, PatchHunk};

/// Arguments for the patch tool (JSON format).
#[derive(Debug, Deserialize)]
struct JsonPatchArgs {
    /// List of file operations.
    patches: Vec<JsonPatchHunk>,
}

/// A single patch hunk in JSON format.
#[derive(Debug, Deserialize)]
struct JsonPatchHunk {
    /// File path.
    path: String,
    /// Operation type: "add", "update", or "delete".
    operation: String,
    /// Content for add/update operations.
    content: Option<String>,
    /// For update: context lines to find the location.
    context: Option<String>,
    /// For update: lines to remove.
    remove: Option<String>,
    /// For update: new file path (rename/move).
    new_path: Option<String>,
}

/// Create the patch tool definition.
#[must_use]
pub fn patch_tool() -> ToolDefinition {
    ToolDefinition {
        name: "patch",
        description: "Apply file modifications. Accepts JSON format with patches array, \
                      or freeform text format starting with '*** Begin Patch'.",
        params: vec![ToolParam {
            name: "patches",
            description: "Array of patch operations (JSON) or freeform patch text.",
            param_type: "array",
            items: Some(crate::tools::definition::ArrayItemType {
                item_type: "object",
            }),
        }],
        required: vec!["patches"],
        executor: Arc::new(execute_patch),
    }
}

fn execute_patch(ctx: &ToolContext, args: Value) -> Result<String, ToolError> {
    // Try to detect format
    let patch = if let Some(text) = args.as_str() {
        // Freeform text format
        parse_freeform_patch(text)?
    } else if let Some(obj) = args.as_object() {
        // Check if it's wrapped in an object with patches key
        if let Some(patches) = obj.get("patches") {
            parse_json_patches(patches)?
        } else {
            return Err(ToolError::InvalidArgs(
                "Expected 'patches' array or freeform text".to_string(),
            ));
        }
    } else if let Some(arr) = args.as_array() {
        // Direct array of patches
        parse_json_patches(&Value::Array(arr.clone()))?
    } else {
        return Err(ToolError::InvalidArgs("Invalid patch format".to_string()));
    };

    // Apply the patch
    apply_patch(&patch, ctx)
}

fn parse_json_patches(value: &Value) -> Result<Patch, ToolError> {
    let args: JsonPatchArgs = serde_json::from_value(serde_json::json!({ "patches": value }))
        .map_err(|e| ToolError::InvalidArgs(format!("Invalid JSON patch format: {e}")))?;

    let hunks = args
        .patches
        .into_iter()
        .map(convert_json_hunk)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Patch { hunks })
}

fn convert_json_hunk(hunk: JsonPatchHunk) -> Result<PatchHunk, ToolError> {
    match hunk.operation.as_str() {
        "add" => {
            let content = hunk.content.ok_or_else(|| {
                ToolError::InvalidArgs("'add' operation requires 'content'".to_string())
            })?;
            Ok(PatchHunk::Add {
                path: hunk.path,
                content,
            })
        }
        "delete" => Ok(PatchHunk::Delete { path: hunk.path }),
        "update" => {
            let content = hunk.content.unwrap_or_default();
            Ok(PatchHunk::Update {
                path: hunk.path,
                new_path: hunk.new_path,
                context: hunk.context,
                remove: hunk.remove,
                add: content,
            })
        }
        op => Err(ToolError::InvalidArgs(format!(
            "Unknown operation: {op}. Use 'add', 'update', or 'delete'"
        ))),
    }
}
