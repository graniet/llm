//! Module for chaining multiple LLM backends in a single prompt sequence.
//! Each step can reference a distinct provider_id ("openai", "anthro", etc.).

use std::collections::HashMap;

use crate::{
    chat::{ChatMessage, ChatRole, MessageType},
    completion::CompletionRequest,
    error::LLMError,
    LLMProvider,
};

#[cfg(feature = "api")]
use crate::api::Server;

/// Stores multiple LLM backends (OpenAI, Anthropic, etc.) identified by a key
#[derive(Default)]
pub struct LLMRegistry {
    pub backends: HashMap<String, Box<dyn LLMProvider>>,
}

impl LLMRegistry {
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Inserts a backend under an identifier, e.g. "openai"
    pub fn insert(&mut self, id: impl Into<String>, llm: Box<dyn LLMProvider>) {
        self.backends.insert(id.into(), llm);
    }

    /// Retrieves a backend by its identifier
    pub fn get(&self, id: &str) -> Option<&dyn LLMProvider> {
        self.backends.get(id).map(|b| b.as_ref())
    }

    #[cfg(feature = "api")]
    /// Starts a REST API server on the specified address
    pub async fn serve(self, addr: impl Into<String>) -> Result<(), LLMError> {
        let server = Server::new(self);
        server.run(&addr.into()).await?;

        Ok(())
    }
}

/// Builder pattern for LLMRegistry
#[derive(Default)]
pub struct LLMRegistryBuilder {
    registry: LLMRegistry,
}

impl LLMRegistryBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a backend under the given id
    pub fn register(mut self, id: impl Into<String>, llm: Box<dyn LLMProvider>) -> Self {
        self.registry.insert(id, llm);
        self
    }

    /// Builds the final LLMRegistry
    pub fn build(self) -> LLMRegistry {
        self.registry
    }
}

/// Response transformation function
type ResponseTransform = Box<dyn Fn(String) -> String + Send + Sync>;

/// Execution mode for a step: Chat or Completion
#[derive(Debug, Clone)]
pub enum MultiChainStepMode {
    Chat,
    Completion,
    SpeechToText,
}

/// Multi-backend chain step
pub struct MultiChainStep {
    provider_id: String,
    id: String,
    template: String,
    mode: MultiChainStepMode,

    // Override parameters
    temperature: Option<f32>,
    max_tokens: Option<u32>,

    // Response transformation
    response_transform: Option<ResponseTransform>,
}

/// Builder for MultiChainStep (Stripe-style)
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

    /// Backend identifier to use, e.g. "openai"
    pub fn provider_id(mut self, pid: impl Into<String>) -> Self {
        self.provider_id = Some(pid.into());
        self
    }

    /// Unique identifier for the step, e.g. "calc1"
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The prompt or template (e.g. "2 * 4 = ?")
    pub fn template(mut self, tmpl: impl Into<String>) -> Self {
        self.template = Some(tmpl.into());
        self
    }

    // Parameters
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

    /// Builds the step
    pub fn build(self) -> Result<MultiChainStep, LLMError> {
        let provider_id = self
            .provider_id
            .ok_or_else(|| LLMError::InvalidRequest("No provider_id set".into()))?;
        let id = self
            .id
            .ok_or_else(|| LLMError::InvalidRequest("No step id set".into()))?;
        let tmpl = self
            .template
            .ok_or_else(|| LLMError::InvalidRequest("No template set".into()))?;

        Ok(MultiChainStep {
            provider_id,
            id,
            template: tmpl,
            mode: self.mode,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            response_transform: self.response_transform,
        })
    }
}

/// The multi-backend chain
pub struct MultiPromptChain<'a> {
    registry: &'a LLMRegistry,
    steps: Vec<MultiChainStep>,
    memory: HashMap<String, String>, // stores responses
}

impl<'a> MultiPromptChain<'a> {
    pub fn new(registry: &'a LLMRegistry) -> Self {
        Self {
            registry,
            steps: vec![],
            memory: HashMap::new(),
        }
    }

    /// Adds a step
    pub fn step(mut self, step: MultiChainStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Executes all steps
    pub async fn run(mut self) -> Result<HashMap<String, String>, LLMError> {
        for step in &self.steps {
            // 1) Replace {{xyz}} in template with existing memory
            let prompt_text = self.replace_template(&step.template);

            // 2) Get the right backend
            let llm = self.registry.get(&step.provider_id).ok_or_else(|| {
                LLMError::InvalidRequest(format!(
                    "No provider with id '{}' found in registry",
                    step.provider_id
                ))
            })?;

            // 3) Execute
            let mut response = match step.mode {
                MultiChainStepMode::Chat => {
                    let messages = vec![ChatMessage {
                        role: ChatRole::User,
                        message_type: MessageType::Text,
                        content: prompt_text,
                    }];
                    llm.chat(&messages).await?.text().unwrap_or_default()
                }
                MultiChainStepMode::Completion => {
                    let mut req = CompletionRequest::new(prompt_text);
                    req.temperature = step.temperature;
                    req.max_tokens = step.max_tokens;
                    let c = llm.complete(&req).await?;
                    c.text.to_string()
                }
                MultiChainStepMode::SpeechToText => llm.transcribe_file(&prompt_text).await?,
            };

            if let Some(transform) = &step.response_transform {
                response = transform(response);
            }

            // 4) Store the response
            self.memory.insert(step.id.clone(), response);
        }
        Ok(self.memory)
    }

    fn replace_template(&self, input: &str) -> String {
        let mut out = input.to_string();
        for (k, v) in &self.memory {
            let pattern = format!("{{{{{k}}}}}");
            out = out.replace(&pattern, v);
        }
        out
    }

    /// Adds multiple steps at once
    pub fn chain(mut self, steps: Vec<MultiChainStep>) -> Self {
        self.steps.extend(steps);
        self
    }
}
