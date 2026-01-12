use async_trait::async_trait;

use crate::{
    embedding::EmbeddingProvider,
    models::{ModelListRequest, ModelListResponse, ModelsProvider},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};

use super::wrapper::ValidatedLLM;

impl LLMProvider for ValidatedLLM {}

#[async_trait]
impl EmbeddingProvider for ValidatedLLM {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, crate::error::LLMError> {
        self.inner().embed(input).await
    }
}

#[async_trait]
impl SpeechToTextProvider for ValidatedLLM {
    async fn transcribe(&self, audio: Vec<u8>) -> Result<String, crate::error::LLMError> {
        self.inner().transcribe(audio).await
    }
}

#[async_trait]
impl TextToSpeechProvider for ValidatedLLM {
    async fn speech(&self, text: &str) -> Result<Vec<u8>, crate::error::LLMError> {
        self.inner().speech(text).await
    }
}

#[async_trait]
impl ModelsProvider for ValidatedLLM {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, crate::error::LLMError> {
        self.inner().list_models(request).await
    }
}
