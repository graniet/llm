use super::llm_builder::LLMBuilder;

impl LLMBuilder {
    /// Set the API version (Azure OpenAI).
    pub fn api_version(mut self, api_version: impl Into<String>) -> Self {
        self.state.api_version = Some(api_version.into());
        self
    }

    /// Set the deployment id (Azure OpenAI).
    pub fn deployment_id(mut self, deployment_id: impl Into<String>) -> Self {
        self.state.deployment_id = Some(deployment_id.into());
        self
    }
}
