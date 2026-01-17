use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use llm::ToolCall;

use super::id::MessageId;
use crate::dialogue::{ParticipantColor, ParticipantId};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum MessageRole {
    User,
    Assistant,
    Tool,
    Error,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum MessageState {
    Pending,
    Streaming,
    Complete,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageKind {
    Text(String),
    ToolCall(ToolInvocation),
    ToolResult(ToolResult),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub id: String,
    pub name: String,
    pub arguments: String,
    pub partial: bool,
}

impl ToolInvocation {
    pub fn from_call(call: &ToolCall) -> Self {
        Self {
            id: call.id.clone(),
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
            partial: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub id: String,
    pub name: String,
    pub output: String,
    pub success: bool,
}

impl ToolResult {
    pub fn success(id: String, name: String, output: String) -> Self {
        Self {
            id,
            name,
            output,
            success: true,
        }
    }

    pub fn failure(id: String, name: String, output: String) -> Self {
        Self {
            id,
            name,
            output,
            success: false,
        }
    }

    pub fn as_tool_call(&self) -> ToolCall {
        ToolCall {
            id: self.id.clone(),
            call_type: "function".to_string(),
            function: llm::FunctionCall {
                name: self.name.clone(),
                arguments: self.output.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageMetadata {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub latency_ms: Option<u128>,
    pub tokens: Option<llm::chat::Usage>,
    /// Participant ID for multi-LLM dialogues.
    #[serde(default)]
    pub participant_id: Option<ParticipantId>,
    /// Participant display name for multi-LLM dialogues.
    #[serde(default)]
    pub participant_name: Option<String>,
    /// Participant color for multi-LLM dialogues.
    #[serde(default)]
    pub participant_color: Option<ParticipantColor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub id: MessageId,
    pub role: MessageRole,
    pub kind: MessageKind,
    pub state: MessageState,
    pub timestamp: DateTime<Utc>,
    pub metadata: MessageMetadata,
    pub version: u64,
}

impl ConversationMessage {
    pub fn new(role: MessageRole, kind: MessageKind) -> Self {
        Self {
            id: MessageId::new(),
            role,
            kind,
            state: MessageState::Pending,
            timestamp: Utc::now(),
            metadata: MessageMetadata::default(),
            version: 0,
        }
    }

    pub fn bump_version(&mut self) {
        self.version = self.version.saturating_add(1);
    }

    pub fn update_text(&mut self, delta: &str) {
        if let MessageKind::Text(text) = &mut self.kind {
            text.push_str(delta);
            self.bump_version();
        }
    }

    pub fn replace_kind(&mut self, kind: MessageKind) {
        self.kind = kind;
        self.bump_version();
    }
}
