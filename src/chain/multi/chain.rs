use std::collections::HashMap;

use crate::{
    chat::{ChatMessage, ChatRole, MessageType},
    completion::CompletionRequest,
    error::LLMError,
    LLMProvider,
};

use super::registry::LLMRegistry;
use super::step::{MultiChainStep, MultiChainStepMode, ResponseTransform};

/// The multi-backend chain.
pub struct MultiPromptChain<'a> {
    registry: &'a LLMRegistry,
    steps: Vec<MultiChainStep>,
    memory: HashMap<String, String>,
}

impl<'a> MultiPromptChain<'a> {
    pub fn new(registry: &'a LLMRegistry) -> Self {
        Self {
            registry,
            steps: vec![],
            memory: HashMap::new(),
        }
    }

    /// Adds a step.
    pub fn step(mut self, step: MultiChainStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Adds multiple steps at once.
    pub fn chain(mut self, steps: Vec<MultiChainStep>) -> Self {
        self.steps.extend(steps);
        self
    }

    /// Executes all steps.
    pub async fn run(mut self) -> Result<HashMap<String, String>, LLMError> {
        for step in &self.steps {
            let response = self.run_step(step).await?;
            self.memory.insert(step.id.clone(), response);
        }
        Ok(self.memory)
    }

    async fn run_step(&self, step: &MultiChainStep) -> Result<String, LLMError> {
        let prompt_text = self.replace_template(&step.template);
        let llm = self.provider(step)?;
        let response = match step.mode {
            MultiChainStepMode::Chat => run_chat(llm, prompt_text).await?,
            MultiChainStepMode::Completion => run_completion(llm, step, prompt_text).await?,
            MultiChainStepMode::SpeechToText => llm.transcribe_file(&prompt_text).await?,
        };
        Ok(apply_transform(response, step.response_transform.as_ref()))
    }

    fn provider(&self, step: &MultiChainStep) -> Result<&dyn LLMProvider, LLMError> {
        self.registry.get(&step.provider_id).ok_or_else(|| {
            LLMError::InvalidRequest(format!(
                "No provider with id '{}' found in registry",
                step.provider_id
            ))
        })
    }

    fn replace_template(&self, input: &str) -> String {
        let mut out = input.to_string();
        for (k, v) in &self.memory {
            let pattern = format!("{{{{{k}}}}}");
            out = out.replace(&pattern, v);
        }
        out
    }
}

async fn run_chat(llm: &dyn LLMProvider, prompt_text: String) -> Result<String, LLMError> {
    let messages = vec![ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: prompt_text,
    }];
    Ok(llm.chat(&messages).await?.text().unwrap_or_default())
}

async fn run_completion(
    llm: &dyn LLMProvider,
    step: &MultiChainStep,
    prompt_text: String,
) -> Result<String, LLMError> {
    let mut req = CompletionRequest::new(prompt_text);
    req.temperature = step.temperature;
    req.max_tokens = step.max_tokens;
    let response = llm.complete(&req).await?;
    Ok(response.text.to_string())
}

fn apply_transform(response: String, transform: Option<&ResponseTransform>) -> String {
    match transform {
        Some(transform) => transform(response),
        None => response,
    }
}
