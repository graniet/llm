use super::llm_builder::LLMBuilder;
use crate::chat::{
    FunctionTool, ParameterProperty, ParametersSchema, StructuredOutputFormat, Tool, ToolChoice,
};
use std::collections::HashMap;

impl LLMBuilder {
    /// Sets the JSON schema for structured output.
    pub fn schema(mut self, schema: impl Into<StructuredOutputFormat>) -> Self {
        self.state.json_schema = Some(schema.into());
        self
    }

    /// Adds a function tool to the builder.
    pub fn function(mut self, function_builder: FunctionBuilder) -> Self {
        if self.state.tools.is_none() {
            self.state.tools = Some(Vec::new());
        }
        if let Some(tools) = &mut self.state.tools {
            tools.push(function_builder.build());
        }
        self
    }

    /// Enable parallel tool use.
    pub fn enable_parallel_tool_use(mut self, enable: bool) -> Self {
        self.state.enable_parallel_tool_use = Some(enable);
        self
    }

    /// Set tool choice.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.state.tool_choice = Some(choice);
        self
    }

    /// Explicitly disable the use of tools, even if they are provided.
    pub fn disable_tools(mut self) -> Self {
        self.state.tool_choice = Some(ToolChoice::None);
        self
    }

    /// Set extra body JSON for compatible providers.
    pub fn extra_body(mut self, extra_body: impl serde::Serialize) -> Self {
        match serde_json::to_value(extra_body) {
            Ok(value) => self.state.extra_body = Some(value),
            Err(err) => log::warn!("extra_body serialization failed: {err}"),
        }
        self
    }
}

/// Builder for function parameters.
pub struct ParamBuilder {
    name: String,
    property_type: String,
    description: String,
    items: Option<Box<ParameterProperty>>,
    enum_list: Option<Vec<String>>,
}

impl ParamBuilder {
    /// Creates a new parameter builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            property_type: "string".to_string(),
            description: String::new(),
            items: None,
            enum_list: None,
        }
    }

    /// Sets the parameter type.
    pub fn type_of(mut self, type_str: impl Into<String>) -> Self {
        self.property_type = type_str.into();
        self
    }

    /// Sets the parameter description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Sets the array item type for array parameters.
    pub fn items(mut self, item_property: ParameterProperty) -> Self {
        self.items = Some(Box::new(item_property));
        self
    }

    /// Sets the enum values for enum parameters.
    pub fn enum_values(mut self, values: Vec<String>) -> Self {
        self.enum_list = Some(values);
        self
    }

    fn build(self) -> (String, ParameterProperty) {
        (
            self.name,
            ParameterProperty {
                property_type: self.property_type,
                description: self.description,
                items: self.items,
                enum_list: self.enum_list,
            },
        )
    }
}

/// Builder for function tools.
pub struct FunctionBuilder {
    name: String,
    description: String,
    parameters: Vec<ParamBuilder>,
    required: Vec<String>,
    raw_schema: Option<serde_json::Value>,
    cache_control: Option<serde_json::Value>,
}

impl FunctionBuilder {
    /// Creates a new function builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            parameters: Vec::new(),
            required: Vec::new(),
            raw_schema: None,
            cache_control: None,
        }
    }

    /// Sets the function description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Adds a parameter to the function.
    pub fn param(mut self, param: ParamBuilder) -> Self {
        self.parameters.push(param);
        self
    }

    /// Marks parameters as required.
    pub fn required(mut self, param_names: Vec<String>) -> Self {
        self.required = param_names;
        self
    }

    /// Provides a full JSON Schema for the parameters.
    pub fn json_schema(mut self, schema: serde_json::Value) -> Self {
        self.raw_schema = Some(schema);
        self
    }

    /// Sets cache control for this tool (e.g. `json!({"type": "ephemeral"})` for Anthropic prompt caching).
    pub fn cache_control(mut self, cache_control: serde_json::Value) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    /// Builds the function tool.
    fn build(self) -> Tool {
        let FunctionBuilder {
            name,
            description,
            parameters,
            required,
            raw_schema,
            cache_control,
        } = self;

        let parameters = build_parameters(raw_schema, parameters, required);

        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name,
                description,
                parameters,
            },
            cache_control,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_builder_without_cache_control() {
        let tool = FunctionBuilder::new("my_tool")
            .description("A test tool")
            .build();

        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "my_tool");
        assert!(tool.cache_control.is_none());
    }

    #[test]
    fn function_builder_with_cache_control() {
        let tool = FunctionBuilder::new("my_tool")
            .description("A test tool")
            .cache_control(serde_json::json!({"type": "ephemeral"}))
            .build();

        assert_eq!(tool.function.name, "my_tool");
        let cc = tool.cache_control.expect("cache_control should be set");
        assert_eq!(cc, serde_json::json!({"type": "ephemeral"}));
    }

    #[test]
    fn tool_serialization_omits_cache_control_when_none() {
        let tool = FunctionBuilder::new("my_tool")
            .description("desc")
            .build();

        let json = serde_json::to_value(&tool).unwrap();
        assert!(json.get("cache_control").is_none());
    }

    #[test]
    fn tool_serialization_includes_cache_control_when_set() {
        let tool = FunctionBuilder::new("my_tool")
            .description("desc")
            .cache_control(serde_json::json!({"type": "ephemeral"}))
            .build();

        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(
            json.get("cache_control").unwrap(),
            &serde_json::json!({"type": "ephemeral"})
        );
    }
}

fn build_parameters(
    raw_schema: Option<serde_json::Value>,
    parameters: Vec<ParamBuilder>,
    required: Vec<String>,
) -> serde_json::Value {
    if let Some(schema) = raw_schema {
        return schema;
    }

    let mut properties = HashMap::new();
    for param in parameters {
        let (name, prop) = param.build();
        properties.insert(name, prop);
    }

    serde_json::to_value(ParametersSchema {
        schema_type: "object".to_string(),
        properties,
        required,
    })
    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
}
