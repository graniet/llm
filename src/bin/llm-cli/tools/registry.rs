use serde_json::Value;
use std::path::Path;
use std::time::Instant;

use llm::builder::FunctionBuilder;

use crate::config::ToolsConfig;

use super::builtin::builtin_tools;
use super::context::ToolContext;
use super::definition::ToolDefinition;
use super::error::ToolError;
use super::user_tools::UserToolsConfig;

#[derive(Clone)]
pub struct ToolRegistry {
    tools: Vec<ToolDefinition>,
}

impl ToolRegistry {
    pub fn from_config(config: &ToolsConfig) -> Self {
        let mut tools = builtin_tools();
        if !config.enabled.is_empty() {
            tools.retain(|tool| config.enabled.iter().any(|name| name == tool.name));
        }
        Self { tools }
    }

    /// Load and add user-defined tools from a YAML file
    pub fn load_user_tools(&mut self, path: &Path) {
        if let Ok(user_config) = UserToolsConfig::load(path) {
            for user_tool in user_config.tools {
                let definition = user_tool.to_definition();
                // Remove any existing tool with the same name
                self.tools.retain(|t| t.name != definition.name);
                self.tools.push(definition);
            }
        }
    }

    /// Get the list of tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name).collect()
    }

    /// Check if a tool exists
    #[allow(dead_code)]
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name == name)
    }

    pub fn function_builders(&self) -> Vec<FunctionBuilder> {
        self.tools
            .iter()
            .map(|tool| tool.function_builder())
            .collect()
    }

    pub fn execute(
        &self,
        name: &str,
        args_json: &str,
        context: &ToolContext,
    ) -> Result<String, ToolError> {
        let tool = self
            .tools
            .iter()
            .find(|tool| tool.name == name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;
        let args = parse_args(args_json)?;
        validate_allowed_paths(&args, context)?;
        let start = Instant::now();
        let result = (tool.executor)(context, args);
        enforce_timeout(start, context.timeout_ms)?;
        result
    }
}

fn parse_args(raw: &str) -> Result<Value, ToolError> {
    if raw.trim().is_empty() {
        return Ok(Value::Object(serde_json::Map::new()));
    }
    serde_json::from_str(raw).map_err(|err| ToolError::InvalidArgs(err.to_string()))
}

fn validate_allowed_paths(args: &Value, context: &ToolContext) -> Result<(), ToolError> {
    if context.allowed_paths.is_empty() {
        return Ok(());
    }
    let mut paths = Vec::new();
    if let Some(path) = args.get("path").and_then(Value::as_str) {
        paths.push(path);
    }
    if let Some(values) = args.get("paths").and_then(Value::as_array) {
        for value in values {
            if let Some(path) = value.as_str() {
                paths.push(path);
            }
        }
    }
    for path in paths {
        let allowed = context
            .allowed_paths
            .iter()
            .any(|root| path.starts_with(root));
        if !allowed {
            return Err(ToolError::Execution(format!("path not allowed: {path}")));
        }
    }
    Ok(())
}

fn enforce_timeout(start: Instant, timeout_ms: u64) -> Result<(), ToolError> {
    if timeout_ms == 0 {
        return Ok(());
    }
    let elapsed = start.elapsed().as_millis() as u64;
    if elapsed > timeout_ms {
        return Err(ToolError::Execution(format!(
            "tool exceeded timeout of {timeout_ms}ms"
        )));
    }
    Ok(())
}
