use async_trait::async_trait;

use crate::{
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
};

use super::wrapper::ResilientLLM;

#[async_trait]
impl CompletionProvider for ResilientLLM {
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        self.retry(|| self.inner.complete(req)).await
    }
}

#[async_trait]
impl EmbeddingProvider for ResilientLLM {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        self.retry(|| self.inner.embed(input.clone())).await
    }
}

#[async_trait]
impl SpeechToTextProvider for ResilientLLM {
    async fn transcribe(&self, audio: Vec<u8>) -> Result<String, LLMError> {
        self.retry(|| self.inner.transcribe(audio.clone())).await
    }
}

#[async_trait]
impl TextToSpeechProvider for ResilientLLM {
    async fn speech(&self, text: &str) -> Result<Vec<u8>, LLMError> {
        let text = text.to_string();
        self.retry(|| self.inner.speech(text.as_str())).await
    }
}

#[async_trait]
impl ModelsProvider for ResilientLLM {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        self.retry(|| self.inner.list_models(request)).await
    }
}
