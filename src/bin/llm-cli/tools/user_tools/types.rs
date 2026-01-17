//! Types for user-defined tools.

use serde::{Deserialize, Serialize};

/// A user-defined tool that can be persisted to YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserTool {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub params: Vec<UserToolParam>,
    pub command: String,
}

/// Parameter definition for a user tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserToolParam {
    pub name: String,
    #[serde(default = "default_param_type")]
    pub param_type: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub required: bool,
}

fn default_param_type() -> String {
    "string".to_string()
}

/// Create a UserTool from the wizard draft.
impl From<crate::runtime::UserToolDraft> for UserTool {
    fn from(draft: crate::runtime::UserToolDraft) -> Self {
        UserTool {
            name: draft.name,
            description: draft.description,
            params: draft
                .params
                .into_iter()
                .map(|p| UserToolParam {
                    name: p.name,
                    param_type: p.param_type,
                    description: p.description,
                    required: p.required,
                })
                .collect(),
            command: draft.command,
        }
    }
}
