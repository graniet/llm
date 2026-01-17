//! Rollback tool handler.
//!
//! Provides the ability to rollback file changes made by tools.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::tools::context::ToolContext;
use crate::tools::definition::{ToolDefinition, ToolParam};
use crate::tools::diff_tracker::DiffTracker;
use crate::tools::error::ToolError;

/// Create the rollback tool definition.
pub fn rollback_tool(tracker: Arc<DiffTracker>) -> ToolDefinition {
    ToolDefinition {
        name: "rollback",
        description: "Rollback file changes made by tools. Can rollback the last N change groups or show a summary of changes.",
        params: vec![
            ToolParam {
                name: "action",
                description: "Action to perform: 'rollback' to undo changes, 'summary' to list changes, 'clear' to discard tracking.",
                param_type: "string",
                items: None,
            },
            ToolParam {
                name: "count",
                description: "Number of change groups to rollback (default: 1). Only used with 'rollback' action.",
                param_type: "number",
                items: None,
            },
        ],
        required: vec!["action"],
        executor: Arc::new(move |ctx, args| {
            let tracker = Arc::clone(&tracker);
            exec_rollback(ctx, args, tracker)
        }),
    }
}

fn exec_rollback(
    _ctx: &ToolContext,
    args: Value,
    tracker: Arc<DiffTracker>,
) -> Result<String, ToolError> {
    let action = args
        .get("action")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::InvalidArgs("missing 'action' parameter".to_string()))?;

    let count = args
        .get("count")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(1);

    match action {
        "rollback" => {
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async { tracker.rollback(count).await })
            });

            match result {
                Ok(rollback_result) => {
                    let output = json!({
                        "success": rollback_result.is_success(),
                        "rolled_back": rollback_result.rolled_back_groups,
                        "restored_files": rollback_result.restored_files.iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>(),
                        "errors": rollback_result.errors.iter()
                            .map(|(p, e)| json!({
                                "file": p.display().to_string(),
                                "error": e
                            }))
                            .collect::<Vec<_>>(),
                    });
                    Ok(serde_json::to_string_pretty(&output).unwrap())
                }
                Err(e) => Err(ToolError::RespondToModel(format!("Rollback failed: {e}"))),
            }
        }
        "summary" => {
            let summary = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async { tracker.summary().await })
            });
            if summary.is_empty() {
                return Ok(json!({"message": "No changes tracked", "changes": []}).to_string());
            }
            let changes: Vec<Value> = summary
                .iter()
                .map(|s| json!({"index": s.index, "tool": s.tool_name, "description": s.description, "file_count": s.file_count}))
                .collect();
            Ok(serde_json::to_string_pretty(
                &json!({"total_groups": summary.len(), "changes": changes}),
            )
            .unwrap())
        }
        "clear" => {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async { tracker.clear().await })
            });
            Ok(json!({
                "success": true,
                "message": "Change tracking cleared"
            })
            .to_string())
        }
        _ => Err(ToolError::InvalidArgs(format!(
            "Unknown action: '{}'. Use 'rollback', 'summary', or 'clear'.",
            action
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_rollback_tool_summary_empty() {
        let tracker = Arc::new(DiffTracker::new(10));
        let ctx = ToolContext::new("/tmp".to_string());
        let args = json!({ "action": "summary" });

        let result = exec_rollback(&ctx, args, tracker).unwrap();
        assert!(result.contains("No changes tracked"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_rollback_tool_rollback_no_changes() {
        let tracker = Arc::new(DiffTracker::new(10));
        let ctx = ToolContext::new("/tmp".to_string());
        let args = json!({ "action": "rollback" });

        let result = exec_rollback(&ctx, args, tracker);
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_rollback_tool_clear() {
        let tracker = Arc::new(DiffTracker::new(10));
        tracker
            .record_create("/tmp/test.txt", "patch", "test")
            .await;

        let ctx = ToolContext::new("/tmp".to_string());
        let args = json!({ "action": "clear" });

        let result = exec_rollback(&ctx, args, Arc::clone(&tracker)).unwrap();
        assert!(result.contains("cleared"));

        let count = tracker.group_count().await;
        assert_eq!(count, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_rollback_tool_full_cycle() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");

        // Create file
        fs::write(&file_path, "content").unwrap();

        // Track and modify
        let tracker = Arc::new(DiffTracker::new(10));
        tracker
            .record_modify(&file_path, "original", "patch", "modify file")
            .await;

        // Verify summary
        let ctx = ToolContext::new(dir.path().to_string_lossy().to_string());
        let summary_result =
            exec_rollback(&ctx, json!({ "action": "summary" }), Arc::clone(&tracker)).unwrap();
        assert!(summary_result.contains("patch"));

        // Rollback
        let rollback_result = exec_rollback(
            &ctx,
            json!({ "action": "rollback", "count": 1 }),
            Arc::clone(&tracker),
        )
        .unwrap();
        assert!(rollback_result.contains("success"));

        // Verify file was restored
        let content = fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "original");
    }
}
