use async_trait::async_trait;

use crate::{
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};

use super::wrapper::ChatWithMemory;

#[async_trait]
impl CompletionProvider for ChatWithMemory {
    async fn complete(
        &self,
        req: &CompletionRequest,
    ) -> Result<CompletionResponse, crate::error::LLMError> {
        self.provider.complete(req).await
    }
}

#[async_trait]
impl EmbeddingProvider for ChatWithMemory {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, crate::error::LLMError> {
        self.provider.embed(input).await
    }
}

#[async_trait]
impl SpeechToTextProvider for ChatWithMemory {
    async fn transcribe(&self, audio: Vec<u8>) -> Result<String, crate::error::LLMError> {
        let provider = self.stt_provider.as_ref().unwrap_or(&self.provider);
        provider.transcribe(audio).await
    }
}

#[async_trait]
impl TextToSpeechProvider for ChatWithMemory {
    async fn speech(&self, text: &str) -> Result<Vec<u8>, crate::error::LLMError> {
        self.provider.speech(text).await
    }
}

#[async_trait]
impl ModelsProvider for ChatWithMemory {}

impl LLMProvider for ChatWithMemory {
    fn tools(&self) -> Option<&[crate::chat::Tool]> {
        self.provider.tools()
    }
}
