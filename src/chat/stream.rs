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
    /// Why the stream stopped, on its last choice: `stop`, `length`,
    /// `tool_calls` or `content_filter`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
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

    /// Token usage of the whole response, when the provider reports it in
    /// its stream (Anthropic's `message_delta`, Bedrock's metadata event).
    /// Sent before `Done`, so a consumer that stops at `Done` still sees it.
    Usage(Usage),

    /// Stream ended with stop reason
    Done {
        /// The reason the stream stopped (e.g., "end_turn", "tool_use")
        stop_reason: String,
    },
}

/// A provider's stop reason in OpenAI's words; one it has no word for is
/// passed on as it came.
pub fn finish_reason_of(stop_reason: &str) -> String {
    match stop_reason {
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        "content_filtered" | "guardrail_intervened" | "refusal" => "content_filter",
        other => other,
    }
    .to_string()
}

impl StreamResponse {
    /// The OpenAI-shaped response of one tool-stream chunk: text and complete
    /// tool calls as deltas, the stop as a `finish_reason`, the usage as a
    /// choice-less response (the last chunk OpenAI sends with
    /// `include_usage`); nothing for a tool call's start and input deltas.
    pub fn from_chunk(chunk: StreamChunk) -> Option<StreamResponse> {
        let delta =
            |content: Option<String>, tool_calls: Option<Vec<ToolCall>>, finish: Option<String>| {
                Some(StreamResponse {
                    choices: vec![StreamChoice {
                        delta: StreamDelta {
                            content,
                            tool_calls,
                        },
                        finish_reason: finish,
                    }],
                    usage: None,
                })
            };
        match chunk {
            StreamChunk::Text(text) => delta(Some(text), None, None),
            StreamChunk::ToolUseComplete { tool_call, .. } => {
                delta(None, Some(vec![tool_call]), None)
            }
            StreamChunk::Done { stop_reason } => {
                delta(None, None, Some(finish_reason_of(&stop_reason)))
            }
            StreamChunk::Usage(usage) => Some(StreamResponse {
                choices: vec![],
                usage: Some(usage),
            }),
            StreamChunk::ToolUseStart { .. } | StreamChunk::ToolUseInputDelta { .. } => None,
        }
    }
}
