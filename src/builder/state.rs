use secrecy::SecretString;

use crate::{
    chat::{StructuredOutputFormat, Tool, ToolChoice},
    memory::MemoryProvider,
};

use super::{backend::LLMBackend, validation::ValidatorFn};

const DEFAULT_VALIDATOR_ATTEMPTS: usize = 3;

#[derive(Default)]
pub(crate) struct BuilderState {
    pub(crate) backend: Option<LLMBackend>,
    pub(crate) api_key: Option<SecretString>,
    pub(crate) base_url: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) temperature: Option<f32>,
    pub(crate) system: Option<String>,
    pub(crate) timeout_seconds: Option<u64>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) embedding_encoding_format: Option<String>,
    pub(crate) embedding_dimensions: Option<u32>,
    pub(crate) validator: Option<Box<ValidatorFn>>,
    pub(crate) validator_attempts: usize,
    pub(crate) tools: Option<Vec<Tool>>,
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) enable_parallel_tool_use: Option<bool>,
    pub(crate) normalize_response: Option<bool>,
    pub(crate) reasoning: Option<bool>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) reasoning_budget_tokens: Option<u32>,
    pub(crate) json_schema: Option<StructuredOutputFormat>,
    pub(crate) api_version: Option<String>,
    pub(crate) deployment_id: Option<String>,
    pub(crate) voice: Option<String>,
    pub(crate) extra_body: Option<serde_json::Value>,
    pub(crate) xai_search_mode: Option<String>,
    pub(crate) xai_search_source_type: Option<String>,
    pub(crate) xai_search_excluded_websites: Option<Vec<String>>,
    pub(crate) xai_search_max_results: Option<u32>,
    pub(crate) xai_search_from_date: Option<String>,
    pub(crate) xai_search_to_date: Option<String>,
    pub(crate) memory: Option<Box<dyn MemoryProvider>>,
    pub(crate) openai_enable_web_search: Option<bool>,
    pub(crate) openai_web_search_context_size: Option<String>,
    pub(crate) openai_web_search_user_location_type: Option<String>,
    pub(crate) openai_web_search_user_location_approximate_country: Option<String>,
    pub(crate) openai_web_search_user_location_approximate_city: Option<String>,
    pub(crate) openai_web_search_user_location_approximate_region: Option<String>,
    pub(crate) resilient_enable: Option<bool>,
    pub(crate) resilient_attempts: Option<usize>,
    pub(crate) resilient_base_delay_ms: Option<u64>,
    pub(crate) resilient_max_delay_ms: Option<u64>,
    pub(crate) resilient_jitter: Option<bool>,
}

impl BuilderState {
    pub(crate) fn new() -> Self {
        Self {
            validator_attempts: DEFAULT_VALIDATOR_ATTEMPTS,
            ..Self::default()
        }
    }
}
