use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use async_trait::async_trait;

use crate::{
    chat::{ChatMessage, ChatProvider, ChatRole, Tool},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    memory::{ChatWithMemory, ChatWithMemoryConfig, MemoryProvider, SlidingWindowMemory},
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};

const RESPONSE_TEXT: &str = "ok";
const TRANSCRIPT: &str = "transcribed";

#[derive(Clone)]
struct RecordingProvider {
    messages: Arc<Mutex<Vec<ChatMessage>>>,
    transcript: String,
    response: String,
    transcribe_calls: Arc<AtomicUsize>,
}

impl RecordingProvider {
    fn new(
        messages: Arc<Mutex<Vec<ChatMessage>>>,
        transcribe_calls: Arc<AtomicUsize>,
        transcript: &str,
        response: &str,
    ) -> Self {
        Self {
            messages,
            transcript: transcript.to_string(),
            response: response.to_string(),
            transcribe_calls,
        }
    }
}

fn unsupported() -> LLMError {
    LLMError::ProviderError("unsupported".to_string())
}

#[async_trait]
impl ChatProvider for RecordingProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
    ) -> Result<Box<dyn crate::chat::ChatResponse>, LLMError> {
        let mut guard = self.messages.lock().expect("messages lock");
        guard.clear();
        guard.extend_from_slice(messages);
        Ok(Box::new(CompletionResponse {
            text: self.response.clone(),
        }))
    }
}

#[async_trait]
impl CompletionProvider for RecordingProvider {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Err(unsupported())
    }
}

#[async_trait]
impl EmbeddingProvider for RecordingProvider {
    async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(unsupported())
    }
}

#[async_trait]
impl SpeechToTextProvider for RecordingProvider {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        self.transcribe_calls.fetch_add(1, Ordering::Relaxed);
        Ok(self.transcript.clone())
    }
}

#[async_trait]
impl TextToSpeechProvider for RecordingProvider {}

#[async_trait]
impl ModelsProvider for RecordingProvider {}

impl LLMProvider for RecordingProvider {}

#[tokio::test]
async fn audio_messages_are_transcribed_before_chat() {
    let messages = Arc::new(Mutex::new(Vec::new()));
    let transcribe_calls = Arc::new(AtomicUsize::new(0));
    let provider = RecordingProvider::new(
        Arc::clone(&messages),
        Arc::clone(&transcribe_calls),
        TRANSCRIPT,
        RESPONSE_TEXT,
    );
    let memory: Arc<tokio::sync::RwLock<Box<dyn MemoryProvider>>> = Arc::new(
        tokio::sync::RwLock::new(Box::new(SlidingWindowMemory::new(5))),
    );
    let config = ChatWithMemoryConfig::new(Arc::new(provider), Arc::clone(&memory));
    let wrapper = ChatWithMemory::with_config(config);

    let msg = ChatMessage::user().audio(vec![1, 2, 3]).build();
    let response = wrapper.chat(&[msg]).await.expect("chat");
    assert_eq!(response.text(), Some(RESPONSE_TEXT.to_string()));
    assert_eq!(transcribe_calls.load(Ordering::Relaxed), 1);

    let recorded = messages.lock().expect("messages lock").clone();
    let recorded_msg = recorded
        .iter()
        .find(|m| matches!(m.role, ChatRole::User))
        .expect("recorded user message");
    assert_eq!(recorded_msg.content, TRANSCRIPT);
    assert!(!recorded_msg.has_audio());

    let stored = memory.read().await.recall("", None).await.expect("recall");
    let stored_msg = stored
        .iter()
        .find(|m| matches!(m.role, ChatRole::User))
        .expect("stored user message");
    assert_eq!(stored_msg.content, TRANSCRIPT);
    assert!(!stored_msg.has_audio());
}
