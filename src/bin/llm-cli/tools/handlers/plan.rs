//! Task planning tool for tracking progress.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tools::context::ToolContext;
use crate::tools::definition::{ToolDefinition, ToolParam};
use crate::tools::error::ToolError;

/// Task status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
}

/// A single task in the plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanItem {
    /// Description of the step.
    pub step: String,
    /// Current status.
    pub status: TaskStatus,
}

/// Arguments for the plan tool.
#[derive(Debug, Deserialize)]
struct PlanArgs {
    /// Optional explanation of plan changes.
    explanation: Option<String>,
    /// List of plan items.
    plan: Vec<PlanItem>,
}

/// Create the plan tool definition.
#[must_use]
pub fn plan_tool() -> ToolDefinition {
    ToolDefinition {
        name: "plan",
        description: "Update the task plan. Use to track progress on multi-step tasks. \
                      Only one task can be in_progress at a time.",
        params: vec![
            ToolParam::simple(
                "explanation",
                "Optional explanation of what changed in the plan.",
                "string",
            ),
            ToolParam::array(
                "plan",
                "Array of plan items with 'step' (string) and 'status' (pending|in_progress|completed).",
                "object",
            ),
        ],
        required: vec!["plan"],
        executor: Arc::new(execute_plan),
    }
}

fn execute_plan(_ctx: &ToolContext, args: Value) -> Result<String, ToolError> {
    let plan_args: PlanArgs =
        serde_json::from_value(args).map_err(|e| ToolError::InvalidArgs(e.to_string()))?;

    // Validate: only one in_progress allowed
    let in_progress_count = plan_args
        .plan
        .iter()
        .filter(|item| item.status == TaskStatus::InProgress)
        .count();

    if in_progress_count > 1 {
        return Err(ToolError::RespondToModel(
            "Only one task can be in_progress at a time".to_string(),
        ));
    }

    // Format output
    let mut output = String::new();

    if let Some(explanation) = &plan_args.explanation {
        output.push_str(&format!("Plan update: {}\n\n", explanation));
    }

    output.push_str("Current plan:\n");

    for (i, item) in plan_args.plan.iter().enumerate() {
        let status_icon = match item.status {
            TaskStatus::Pending => "[ ]",
            TaskStatus::InProgress => "[→]",
            TaskStatus::Completed => "[✓]",
        };
        output.push_str(&format!("{}. {} {}\n", i + 1, status_icon, item.step));
    }

    // Summary
    let completed = plan_args
        .plan
        .iter()
        .filter(|i| i.status == TaskStatus::Completed)
        .count();
    let total = plan_args.plan.len();

    output.push_str(&format!("\nProgress: {}/{} completed", completed, total));

    Ok(output)
}
