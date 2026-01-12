use std::sync::Arc;

use async_trait::async_trait;

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
        self.reset_cycle_counter(messages);
        self.remember_messages(messages).await?;

        let mut context = self.load_context().await?;
        self.maybe_summarize(&mut context).await?;
        context.extend_from_slice(messages);

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

    async fn load_context(&self) -> Result<Vec<ChatMessage>, LLMError> {
        let mem = self.memory.read().await;
        mem.recall("", None).await
    }

    async fn maybe_summarize(&self, context: &mut Vec<ChatMessage>) -> Result<(), LLMError> {
        if !self.needs_summary().await? {
            return Ok(());
        }
        let summary = self.provider.summarize_history(context).await?;
        self.replace_with_summary(summary).await?;
        *context = self.load_context().await?;
        Ok(())
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
