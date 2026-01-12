use crossterm::event::{KeyEvent, MouseEvent};
use tokio::sync::oneshot;

use crate::conversation::{ConversationId, MessageId, ToolInvocation, ToolResult};

#[derive(Debug)]
pub enum InputEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    Paste(String),
    Resize(u16, u16),
}

#[derive(Debug)]
pub enum AppEvent {
    Input(InputEvent),
    Tick,
    Stream(StreamEvent),
    Tool(ToolEvent),
}

#[derive(Debug)]
pub enum StreamEvent {
    Started {
        conversation_id: ConversationId,
    },
    TextDelta {
        conversation_id: ConversationId,
        message_id: MessageId,
        delta: String,
    },
    ToolCallStart {
        conversation_id: ConversationId,
        call_id: String,
        name: String,
    },
    ToolCallDelta {
        conversation_id: ConversationId,
        call_id: String,
        partial_json: String,
    },
    ToolCallComplete {
        conversation_id: ConversationId,
        invocation: ToolInvocation,
    },
    Usage {
        conversation_id: ConversationId,
        message_id: MessageId,
        usage: llm::chat::Usage,
    },
    Done {
        conversation_id: ConversationId,
    },
    Error {
        conversation_id: ConversationId,
        message_id: MessageId,
        error: String,
    },
}

pub struct ToolApprovalRequest {
    pub conversation_id: ConversationId,
    pub invocation: ToolInvocation,
    pub respond_to: oneshot::Sender<bool>,
}

impl std::fmt::Debug for ToolApprovalRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolApprovalRequest")
            .field("conversation_id", &self.conversation_id)
            .field("invocation", &self.invocation)
            .finish()
    }
}

#[derive(Debug)]
pub enum ToolEvent {
    ApprovalRequested(ToolApprovalRequest),
    Result {
        conversation_id: ConversationId,
        result: ToolResult,
    },
}
