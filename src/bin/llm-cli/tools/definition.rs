use serde_json::Value;
use std::sync::Arc;

use llm::builder::{FunctionBuilder, ParamBuilder};
use llm::chat::ParameterProperty;

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
    /// For array types, the item schema.
    pub items: Option<ArrayItemType>,
}

/// Type definition for array items.
#[derive(Clone)]
pub struct ArrayItemType {
    pub item_type: &'static str,
}

impl ToolParam {
    /// Create a simple parameter.
    pub const fn simple(
        name: &'static str,
        description: &'static str,
        param_type: &'static str,
    ) -> Self {
        Self {
            name,
            description,
            param_type,
            items: None,
        }
    }

    /// Create an array parameter with item type.
    pub const fn array(
        name: &'static str,
        description: &'static str,
        item_type: &'static str,
    ) -> Self {
        Self {
            name,
            description,
            param_type: "array",
            items: Some(ArrayItemType { item_type }),
        }
    }
}

impl ToolDefinition {
    pub fn function_builder(&self) -> FunctionBuilder {
        let mut builder = FunctionBuilder::new(self.name).description(self.description);
        for param in &self.params {
            let mut param_builder = ParamBuilder::new(param.name)
                .description(param.description)
                .type_of(param.param_type);

            // Add items schema for array types
            if let Some(items) = &param.items {
                param_builder = param_builder.items(ParameterProperty {
                    property_type: items.item_type.to_string(),
                    description: String::new(),
                    items: None,
                    enum_list: None,
                });
            }

            builder = builder.param(param_builder);
        }
        if !self.required.is_empty() {
            builder = builder.required(self.required.iter().map(|s| s.to_string()).collect());
        }
        builder
    }
}
