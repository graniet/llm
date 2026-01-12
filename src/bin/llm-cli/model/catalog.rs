use llm::models::ModelListRequest;
use llm::LLMProvider;
use std::sync::Arc;

use crate::config::ModelConfig;

use super::entry::ModelInfo;

#[derive(Debug, Clone)]
pub struct ModelCatalog {
    models: Vec<ModelInfo>,
}

impl ModelCatalog {
    pub fn from_config(models: &[ModelConfig]) -> Self {
        let entries = models
            .iter()
            .map(|cfg| ModelInfo {
                id: cfg.id.clone(),
                context_window: cfg.context_window,
                max_output_tokens: cfg.max_output_tokens,
                pricing: cfg.pricing.clone(),
                supports_tools: cfg.supports_tools,
                supports_vision: cfg.supports_vision,
                supports_streaming: cfg.supports_streaming,
            })
            .collect();
        Self { models: entries }
    }

    pub async fn from_provider(provider: Arc<dyn LLMProvider>) -> Self {
        let response = provider
            .list_models(Some(&ModelListRequest::default()))
            .await;
        let entries = response
            .map(|resp| {
                resp.get_models()
                    .into_iter()
                    .map(ModelInfo::new)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Self { models: entries }
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    pub fn list(&self) -> &[ModelInfo] {
        &self.models
    }
}
