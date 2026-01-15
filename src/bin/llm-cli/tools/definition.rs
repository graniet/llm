use serde_json::Value;
use std::sync::Arc;

use llm::builder::{FunctionBuilder, ParamBuilder};

use super::context::ToolContext;
use super::error::ToolError;

pub type ToolExecutor = Arc<dyn Fn(&ToolContext, Value) -> Result<String, ToolError> + Send + Sync>;

#[derive(Clone)]
pub struct ToolDefinition {
    pub name: &'static str,
    pub description: &'static str,
    pub params: Vec<ToolParam>,
    pub required: Vec<&'static str>,
    pub executor: ToolExecutor,
}

#[derive(Clone)]
pub struct ToolParam {
    pub name: &'static str,
    pub description: &'static str,
    pub param_type: &'static str,
}

impl ToolDefinition {
    pub fn function_builder(&self) -> FunctionBuilder {
        let mut builder = FunctionBuilder::new(self.name).description(self.description);
        for param in &self.params {
            builder = builder.param(
                ParamBuilder::new(param.name)
                    .description(param.description)
                    .type_of(param.param_type),
            );
        }
        if !self.required.is_empty() {
            builder = builder.required(self.required.iter().map(|s| s.to_string()).collect());
        }
        builder
    }
}
