use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use llm::chat::ChatMessage;
use llm::LLMProvider;

use crate::conversation::{ConversationId, MessageId};
use crate::provider::ProviderCapabilities;
use crate::runtime::AppEvent;

use super::runner::run_stream;

#[derive(Clone)]
pub struct StreamRequest {
    pub conversation_id: ConversationId,
    pub message_id: MessageId,
    pub provider: Arc<dyn LLMProvider>,
    pub messages: Vec<ChatMessage>,
    pub capabilities: ProviderCapabilities,
}

pub struct StreamManager {
    sender: mpsc::Sender<AppEvent>,
    active: HashMap<ConversationId, CancellationToken>,
}

impl StreamManager {
    pub fn new(sender: mpsc::Sender<AppEvent>) -> Self {
        Self {
            sender,
            active: HashMap::new(),
        }
    }

    pub fn cancel(&mut self, id: ConversationId) {
        if let Some(token) = self.active.remove(&id) {
            token.cancel();
        }
    }

    pub fn start(&mut self, request: StreamRequest) -> CancellationToken {
        let token = CancellationToken::new();
        self.active.insert(request.conversation_id, token.clone());
        let sender = self.sender.clone();
        tokio::spawn(run_stream(request, sender, token.clone()));
        token
    }
}
