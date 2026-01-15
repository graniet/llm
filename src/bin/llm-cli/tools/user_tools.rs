use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::context::ToolContext;
use super::definition::{ToolDefinition, ToolParam};
use super::error::ToolError;

/// A user-defined tool that can be persisted to YAML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserTool {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub params: Vec<UserToolParam>,
    pub command: String,
}

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

/// Collection of user-defined tools
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
    /// Convert to a ToolDefinition that can be registered in the ToolRegistry
    pub fn to_definition(&self) -> ToolDefinition {
        let params: Vec<ToolParam> = self
            .params
            .iter()
            .map(|p| ToolParam {
                name: Box::leak(p.name.clone().into_boxed_str()),
                description: Box::leak(p.description.clone().into_boxed_str()),
                param_type: Box::leak(p.param_type.clone().into_boxed_str()),
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
            executor: Arc::new(move |ctx, args| execute_command_tool(&command, &tool_name, ctx, args)),
        }
    }
}

fn execute_command_tool(
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

/// Create a UserTool from the wizard draft
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let ctx = ToolContext {
            working_dir: ".".to_string(),
            allowed_paths: vec![],
            timeout_ms: 5000,
        };
        let args = serde_json::json!({
            "name": "world"
        });
        let result = execute_command_tool("echo hello {{name}}", "test", &ctx, args);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("hello world"));
    }
}
