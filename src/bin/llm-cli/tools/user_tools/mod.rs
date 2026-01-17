//! User-defined tools that can be persisted to YAML.

mod executor;
mod types;

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::definition::{ToolDefinition, ToolParam};

pub use executor::execute_command_tool;
pub use types::{UserTool, UserToolParam};

/// Collection of user-defined tools.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserToolsConfig {
    #[serde(default)]
    pub tools: Vec<UserTool>,
}

impl UserToolsConfig {
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_yaml::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, content)
    }

    pub fn add_tool(&mut self, tool: UserTool) {
        // Remove existing tool with same name
        self.tools.retain(|t| t.name != tool.name);
        self.tools.push(tool);
    }

    pub fn remove_tool(&mut self, name: &str) -> bool {
        let len_before = self.tools.len();
        self.tools.retain(|t| t.name != name);
        self.tools.len() < len_before
    }

    pub fn get_tool(&self, name: &str) -> Option<&UserTool> {
        self.tools.iter().find(|t| t.name == name)
    }
}

impl UserTool {
    /// Convert to a ToolDefinition that can be registered in the ToolRegistry.
    pub fn to_definition(&self) -> ToolDefinition {
        let params: Vec<ToolParam> = self
            .params
            .iter()
            .map(|p| ToolParam {
                name: Box::leak(p.name.clone().into_boxed_str()),
                description: Box::leak(p.description.clone().into_boxed_str()),
                param_type: Box::leak(p.param_type.clone().into_boxed_str()),
                items: None,
            })
            .collect();

        let required: Vec<&'static str> = self
            .params
            .iter()
            .filter(|p| p.required)
            .map(|p| -> &'static str { Box::leak(p.name.clone().into_boxed_str()) })
            .collect();

        let command = self.command.clone();
        let tool_name = self.name.clone();

        ToolDefinition {
            name: Box::leak(self.name.clone().into_boxed_str()),
            description: Box::leak(self.description.clone().into_boxed_str()),
            params,
            required,
            executor: Arc::new(move |ctx, args| {
                execute_command_tool(&command, &tool_name, ctx, args)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::context::ToolContext;

    #[test]
    fn serialize_deserialize_user_tool() {
        let tool = UserTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            params: vec![UserToolParam {
                name: "message".to_string(),
                param_type: "string".to_string(),
                description: "The message".to_string(),
                required: true,
            }],
            command: "echo {{message}}".to_string(),
        };

        let yaml = serde_yaml::to_string(&tool).unwrap();
        let parsed: UserTool = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.name, "test_tool");
    }

    #[test]
    fn parse_yaml_config() {
        let content = r#"
tools:
  - name: git_status
    description: Get git status
    command: git status
"#;
        let config: UserToolsConfig = serde_yaml::from_str(content).unwrap();
        assert_eq!(config.tools.len(), 1);
        assert_eq!(config.tools[0].name, "git_status");
    }

    #[test]
    fn command_substitution() {
        let ctx = ToolContext::new(".".to_string()).with_timeout(5000);
        let args = serde_json::json!({
            "name": "world"
        });
        let result = execute_command_tool("echo hello {{name}}", "test", &ctx, args);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("hello world"));
    }
}
