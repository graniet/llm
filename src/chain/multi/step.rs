use crate::error::LLMError;

/// Response transformation function.
pub type ResponseTransform = Box<dyn Fn(String) -> String + Send + Sync>;

/// Execution mode for a step: Chat, Completion, or SpeechToText.
#[derive(Debug, Clone)]
pub enum MultiChainStepMode {
    Chat,
    Completion,
    SpeechToText,
}

/// Multi-backend chain step.
pub struct MultiChainStep {
    pub(crate) provider_id: String,
    pub(crate) id: String,
    pub(crate) template: String,
    pub(crate) mode: MultiChainStepMode,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) response_transform: Option<ResponseTransform>,
}

/// Builder for MultiChainStep (Stripe-style).
pub struct MultiChainStepBuilder {
    provider_id: Option<String>,
    id: Option<String>,
    template: Option<String>,
    mode: MultiChainStepMode,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    response_transform: Option<ResponseTransform>,
}

impl MultiChainStepBuilder {
    pub fn new(mode: MultiChainStepMode) -> Self {
        Self {
            provider_id: None,
            id: None,
            template: None,
            mode,
            temperature: None,
            top_p: None,
            max_tokens: None,
            response_transform: None,
        }
    }

    /// Backend identifier to use, e.g. "openai".
    pub fn provider_id(mut self, pid: impl Into<String>) -> Self {
        self.provider_id = Some(pid.into());
        self
    }

    /// Unique identifier for the step, e.g. "calc1".
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The prompt or template (e.g. "2 * 4 = ?").
    pub fn template(mut self, tmpl: impl Into<String>) -> Self {
        self.template = Some(tmpl.into());
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn max_tokens(mut self, mt: u32) -> Self {
        self.max_tokens = Some(mt);
        self
    }

    pub fn response_transform<F>(mut self, func: F) -> Self
    where
        F: Fn(String) -> String + Send + Sync + 'static,
    {
        self.response_transform = Some(Box::new(func));
        self
    }

    /// Builds the step.
    pub fn build(self) -> Result<MultiChainStep, LLMError> {
        let provider_id = require_field(self.provider_id, "provider_id")?;
        let id = require_field(self.id, "step id")?;
        let template = require_field(self.template, "template")?;

        Ok(MultiChainStep {
            provider_id,
            id,
            template,
            mode: self.mode,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            response_transform: self.response_transform,
        })
    }
}

fn require_field(value: Option<String>, name: &str) -> Result<String, LLMError> {
    value.ok_or_else(|| LLMError::InvalidRequest(format!("No {name} set")))
}
