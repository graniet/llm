use async_trait::async_trait;

use crate::{
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    error::LLMError,
};

use super::{helpers::decrement_attempts, wrapper::ValidatedLLM};

#[async_trait]
impl CompletionProvider for ValidatedLLM {
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let mut remaining_attempts = self.attempts();
        loop {
            let response = self.inner().complete(req).await?;
            match (self.validator())(&response.text) {
                Ok(()) => return Ok(response),
                Err(err) => {
                    remaining_attempts = decrement_attempts(remaining_attempts, &err)?;
                }
            }
        }
    }
}
