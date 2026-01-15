use serde::{Deserialize, Serialize};

use crate::ToolCall;

use super::usage::Usage;

/// Stream response chunk that mimics OpenAI's streaming response format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResponse {
    /// Array of choices in the response
    pub choices: Vec<StreamChoice>,
    /// Usage metadata, typically present in the final chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Individual choice in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    /// Delta containing the incremental content
    pub delta: StreamDelta,
}

/// Delta content in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDelta {
    /// The incremental content, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The incremental tool calls, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// A streaming chunk that can be either text or a tool call event.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Text content delta
    Text(String),

    /// Tool use block started (contains tool id and name)
    ToolUseStart {
        /// The index of this content block in the response
        index: usize,
        /// The unique ID for this tool use
        id: String,
        /// The name of the tool being called
        name: String,
    },

    /// Tool use input JSON delta (partial JSON string)
    ToolUseInputDelta {
        /// The index of this content block
        index: usize,
        /// Partial JSON string for the tool input
        partial_json: String,
    },

    /// Tool use block complete with assembled ToolCall
    ToolUseComplete {
        /// The index of this content block
        index: usize,
        /// The complete tool call with id, name, and parsed arguments
        tool_call: ToolCall,
    },

    /// Stream ended with stop reason
    Done {
        /// The reason the stream stopped (e.g., "end_turn", "tool_use")
        stop_reason: String,
    },
}
