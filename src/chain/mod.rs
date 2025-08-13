mod multi;

use crate::{error::LLMError, LLMProvider};
use std::collections::HashMap;

pub use multi::{
    LLMRegistry, LLMRegistryBuilder, MultiChainStepBuilder, MultiChainStepMode, MultiPromptChain,
};

/// Execution mode for a chain step
#[derive(Debug, Clone)]
pub enum ChainStepMode {
    /// Execute step using chat completion
    Chat,
    /// Execute step using text completion
    Completion,
}

/// Represents a single step in a prompt chain
#[derive(Debug, Clone)]
pub struct ChainStep {
    /// Unique identifier for this step
    pub id: String,
    /// Prompt template with {{variable}} placeholders
    pub template: String,
    /// Execution mode (chat or completion)
    pub mode: ChainStepMode,
    /// Optional temperature parameter (0.0-1.0) controlling randomness
    pub temperature: Option<f32>,
    /// Optional maximum tokens to generate in response
    pub max_tokens: Option<u32>,
    /// Optional top_p parameter for nucleus sampling
    pub top_p: Option<f32>,
}

/// Builder pattern for constructing ChainStep instances
pub struct ChainStepBuilder {
    id: String,
    template: String,
    mode: ChainStepMode,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
}

impl ChainStepBuilder {
    /// Creates a new ChainStepBuilder
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the step
    /// * `template` - Prompt template with {{variable}} placeholders
    /// * `mode` - Execution mode (chat or completion)
    pub fn new(id: impl Into<String>, template: impl Into<String>, mode: ChainStepMode) -> Self {
        Self {
            id: id.into(),
            template: template.into(),
            mode,
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
        }
    }

    /// Sets the temperature parameter
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets the maximum tokens parameter
    pub fn max_tokens(mut self, mt: u32) -> Self {
        self.max_tokens = Some(mt);
        self
    }

    /// Sets the top_p parameter
    pub fn top_p(mut self, val: f32) -> Self {
        self.top_p = Some(val);
        self
    }

    /// Sets the top_k parameter
    pub fn top_k(mut self, val: u32) -> Self {
        self.top_k = Some(val);
        self
    }

    /// Builds and returns a ChainStep instance
    pub fn build(self) -> ChainStep {
        ChainStep {
            id: self.id,
            template: self.template,
            mode: self.mode,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
        }
    }
}

/// Manages a sequence of prompt steps with variable substitution
pub struct PromptChain<'a> {
    llm: &'a dyn LLMProvider,
    steps: Vec<ChainStep>,
    memory: HashMap<String, String>,
}

impl<'a> PromptChain<'a> {
    /// Creates a new PromptChain with the given LLM provider
    pub fn new(llm: &'a dyn LLMProvider) -> Self {
        Self {
            llm,
            steps: Vec::new(),
            memory: HashMap::new(),
        }
    }

    /// Adds a step to the chain
    pub fn step(mut self, step: ChainStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Executes all steps in the chain and returns the results
    pub async fn run(mut self) -> Result<HashMap<String, String>, LLMError> {
        for step in &self.steps {
            let prompt = self.apply_template(&step.template);

            let response_text = match step.mode {
                ChainStepMode::Chat => {
                    let messages = vec![crate::chat::ChatMessage {
                        role: crate::chat::ChatRole::User,
                        message_type: crate::chat::MessageType::Text,
                        content: prompt,
                    }];
                    self.llm.chat(&messages).await?
                }
                ChainStepMode::Completion => {
                    let mut req = crate::completion::CompletionRequest::new(prompt);
                    req.max_tokens = step.max_tokens;
                    req.temperature = step.temperature;
                    let resp = self.llm.complete(&req).await?;
                    Box::new(resp)
                }
            };

            self.memory
                .insert(step.id.clone(), response_text.text().unwrap_or_default());
        }

        Ok(self.memory)
    }

    /// Replaces {{variable}} placeholders in template with values from memory
    fn apply_template(&self, input: &str) -> String {
        let mut result = input.to_string();
        for (k, v) in &self.memory {
            let pattern = format!("{{{{{k}}}}}");
            result = result.replace(&pattern, v);
        }
        result
    }
}
