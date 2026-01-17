use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, Tool},
    error::LLMError,
};

use super::wrapper::ChatWithMemory;

#[async_trait]
impl ChatProvider for ChatWithMemory {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let normalized = self.normalize_messages(messages).await?;
        self.reset_cycle_counter(&normalized);
        self.remember_messages(&normalized).await?;

        let mut context = self.load_context().await?;
        let summarized = self.maybe_summarize(&mut context).await?;
        if summarized {
            context.extend_from_slice(&normalized);
        }

        let response = self.provider.chat_with_tools(&context, tools).await?;
        if let Some(text) = response.text() {
            self.spawn_record_response(text);
        }

        Ok(response)
    }

    async fn memory_contents(&self) -> Option<Vec<ChatMessage>> {
        Some(self.memory_contents().await)
    }
}

impl ChatWithMemory {
    fn reset_cycle_counter(&self, messages: &[ChatMessage]) {
        if messages.iter().any(|m| matches!(m.role, ChatRole::User)) {
            self.cycle_counter
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    async fn remember_messages(&self, messages: &[ChatMessage]) -> Result<(), LLMError> {
        let mut mem = self.memory.write().await;
        for msg in messages {
            mem.remember(msg).await?;
        }
        Ok(())
    }

    async fn normalize_messages(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<ChatMessage>, LLMError> {
        try_join_all(messages.iter().map(|msg| self.normalize_message(msg))).await
    }

    async fn normalize_message(&self, message: &ChatMessage) -> Result<ChatMessage, LLMError> {
        if !message.has_audio() {
            return Ok(message.clone());
        }
        let audio = message
            .audio_data()
            .ok_or_else(|| LLMError::InvalidRequest("Audio payload missing for message".into()))?;
        let transcript = self.transcribe_audio(audio).await?;
        let builder = match message.role {
            ChatRole::User => ChatMessage::user(),
            ChatRole::Assistant => ChatMessage::assistant(),
        };
        Ok(builder.content(transcript).build())
    }

    async fn transcribe_audio(&self, audio: &[u8]) -> Result<String, LLMError> {
        let provider = self.stt_provider.as_ref().unwrap_or(&self.provider);
        provider.transcribe(audio.to_vec()).await
    }

    async fn load_context(&self) -> Result<Vec<ChatMessage>, LLMError> {
        let mem = self.memory.read().await;
        mem.recall("", None).await
    }

    async fn maybe_summarize(&self, context: &mut Vec<ChatMessage>) -> Result<bool, LLMError> {
        if !self.needs_summary().await? {
            return Ok(false);
        }
        let summary = self.provider.summarize_history(context).await?;
        self.replace_with_summary(summary).await?;
        *context = self.load_context().await?;
        Ok(true)
    }

    async fn needs_summary(&self) -> Result<bool, LLMError> {
        let mem = self.memory.read().await;
        Ok(mem.needs_summary())
    }

    async fn replace_with_summary(&self, summary: String) -> Result<(), LLMError> {
        let mut mem = self.memory.write().await;
        mem.replace_with_summary(summary);
        Ok(())
    }

    fn spawn_record_response(&self, text: String) {
        let memory = self.memory.clone();
        let role = self.role.clone();
        tokio::spawn(async move {
            if let Err(err) = persist_response(memory, role, text).await {
                log::warn!("Memory save error: {err}");
            }
        });
    }
}

async fn persist_response(
    memory: Arc<tokio::sync::RwLock<Box<dyn crate::memory::MemoryProvider>>>,
    role: Option<String>,
    text: String,
) -> Result<(), LLMError> {
    let formatted = match &role {
        Some(r) => format!("[{r}] {text}"),
        None => text,
    };
    let msg = ChatMessage::assistant().content(formatted).build();

    let mut mem = memory.write().await;
    match role {
        Some(r) => mem.remember_with_role(&msg, r).await,
        None => mem.remember(&msg).await,
    }
}
