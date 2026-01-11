use std::sync::Arc;
use std::time::Instant;

use futures::future::join_all;

use crate::{
    chat::{ChatMessage, Tool},
    completion::CompletionRequest,
    error::LLMError,
    LLMProvider,
};

use super::types::ParallelEvalResult;
use crate::evaluator::ScoringFn;

/// Evaluator for running multiple LLM providers in parallel and selecting the best response.
pub struct ParallelEvaluator {
    providers: Vec<(String, Box<dyn LLMProvider>)>,
    scoring_fns: Vec<Box<ScoringFn>>,
    include_timing: bool,
}

impl ParallelEvaluator {
    /// Creates a new parallel evaluator.
    pub fn new(providers: Vec<(String, Box<dyn LLMProvider>)>) -> Self {
        Self {
            providers,
            scoring_fns: Vec::new(),
            include_timing: true,
        }
    }

    /// Adds a scoring function to evaluate LLM responses.
    pub fn scoring<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> f32 + Send + Sync + 'static,
    {
        self.scoring_fns.push(Box::new(f));
        self
    }

    /// Sets whether to include timing information in results.
    pub fn include_timing(mut self, include: bool) -> Self {
        self.include_timing = include;
        self
    }

    /// Evaluates chat responses from all providers in parallel for the given messages.
    pub async fn evaluate_chat_parallel(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<ParallelEvalResult>, LLMError> {
        let messages = Arc::new(messages.to_vec());
        self.evaluate_chat(messages, None).await
    }

    /// Evaluates chat responses with tools from all providers in parallel.
    pub async fn evaluate_chat_with_tools_parallel(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Vec<ParallelEvalResult>, LLMError> {
        let messages = Arc::new(messages.to_vec());
        let tools = tools.map(|t| Arc::new(t.to_vec()));
        self.evaluate_chat(messages, tools).await
    }

    /// Evaluates completion responses from all providers in parallel.
    pub async fn evaluate_completion_parallel(
        &self,
        request: &CompletionRequest,
    ) -> Result<Vec<ParallelEvalResult>, LLMError> {
        let request = Arc::new(request.clone());
        let futures = self.providers.iter().map(|(id, provider)| {
            let id = id.clone();
            let request = request.clone();
            async move {
                let start = Instant::now();
                let result = provider.complete(&request).await.map(|r| r.text);
                (id, result, start.elapsed().as_millis())
            }
        });

        Ok(self.collect_results(join_all(futures).await))
    }

    /// Returns the best response based on scoring.
    pub fn best_response<'a>(
        &self,
        results: &'a [ParallelEvalResult],
    ) -> Option<&'a ParallelEvalResult> {
        results.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn collect_results(
        &self,
        results: Vec<(String, Result<String, LLMError>, u128)>,
    ) -> Vec<ParallelEvalResult> {
        let mut eval_results = Vec::new();
        for (id, result, elapsed) in results {
            match result {
                Ok(text) => eval_results.push(self.build_result(id, text, elapsed)),
                Err(err) => log::warn!("Error from provider {id}: {err}"),
            }
        }
        eval_results
    }

    fn build_result(&self, id: String, text: String, elapsed: u128) -> ParallelEvalResult {
        ParallelEvalResult {
            score: self.compute_score(&text),
            time_ms: self.timing_or_zero(elapsed),
            text,
            provider_id: id,
        }
    }

    fn timing_or_zero(&self, elapsed: u128) -> u128 {
        if self.include_timing {
            elapsed
        } else {
            0
        }
    }

    async fn evaluate_chat(
        &self,
        messages: Arc<Vec<ChatMessage>>,
        tools: Option<Arc<Vec<Tool>>>,
    ) -> Result<Vec<ParallelEvalResult>, LLMError> {
        let futures = self.providers.iter().map(|(id, provider)| {
            let id = id.clone();
            let messages = messages.clone();
            let tools = tools.clone();
            async move {
                let start = Instant::now();
                let result = match tools {
                    Some(tools) => provider
                        .chat_with_tools(&messages, Some(&tools))
                        .await
                        .map(|r| r.text().unwrap_or_default()),
                    None => provider
                        .chat(&messages)
                        .await
                        .map(|r| r.text().unwrap_or_default()),
                };
                (id, result, start.elapsed().as_millis())
            }
        });

        Ok(self.collect_results(join_all(futures).await))
    }

    fn compute_score(&self, response: &str) -> f32 {
        self.scoring_fns.iter().map(|sc| sc(response)).sum()
    }
}
