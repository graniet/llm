use llm::builder::{LLMBackend, LLMBuilder};
use llm::LLMProvider;

use crate::provider::error::ProviderBuildError;

use super::ResolvedConfig;

pub(super) fn build_provider(
    backend: &LLMBackend,
    resolved: &ResolvedConfig,
    tool_builders: Vec<llm::builder::FunctionBuilder>,
) -> Result<Box<dyn LLMProvider>, ProviderBuildError> {
    let builder = LLMBuilder::new().backend(backend.clone());
    let builder = resolved.apply(builder);
    let builder = apply_tools(builder, tool_builders);
    builder
        .build()
        .map_err(|err| ProviderBuildError::Build(err.to_string()))
}

impl ResolvedConfig {
    fn apply(&self, mut builder: LLMBuilder) -> LLMBuilder {
        if let Some(value) = self.model.clone() {
            builder = builder.model(value);
        }
        if let Some(value) = self.system.clone() {
            builder = builder.system(value);
        }
        if let Some(value) = self.api_key.clone() {
            builder = builder.api_key(value);
        }
        if let Some(value) = self.base_url.clone() {
            builder = builder.base_url(value);
        }
        if let Some(value) = self.temperature {
            builder = builder.temperature(value);
        }
        if let Some(value) = self.max_tokens {
            builder = builder.max_tokens(value);
        }
        if let Some(value) = self.timeout_seconds {
            builder = builder.timeout_seconds(value);
        }
        builder
    }
}

fn apply_tools(
    mut builder: LLMBuilder,
    tool_builders: Vec<llm::builder::FunctionBuilder>,
) -> LLMBuilder {
    for tool in tool_builders {
        builder = builder.function(tool);
    }
    builder
}
