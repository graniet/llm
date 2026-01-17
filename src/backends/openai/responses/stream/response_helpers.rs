use crate::chat::{StreamChoice, StreamDelta, StreamResponse, Usage};
use crate::{FunctionCall, ToolCall};

use super::events::ToolState;

pub(super) fn tool_call_from_state(state: &ToolState) -> ToolCall {
    tool_call_with_arguments(state, "")
}

pub(super) fn tool_call_with_arguments(state: &ToolState, arguments: &str) -> ToolCall {
    ToolCall {
        id: state.call_id.clone(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: state.name.clone(),
            arguments: arguments.to_string(),
        },
    }
}

pub(super) fn stream_response_text(delta: String) -> StreamResponse {
    StreamResponse {
        choices: vec![StreamChoice {
            delta: StreamDelta {
                content: Some(delta),
                tool_calls: None,
            },
        }],
        usage: None,
    }
}

pub(super) fn stream_response_tool_call(tool_call: ToolCall) -> StreamResponse {
    StreamResponse {
        choices: vec![StreamChoice {
            delta: StreamDelta {
                content: None,
                tool_calls: Some(vec![tool_call]),
            },
        }],
        usage: None,
    }
}

pub(super) fn stream_response_usage(usage: Usage) -> StreamResponse {
    StreamResponse {
        choices: vec![StreamChoice {
            delta: StreamDelta {
                content: None,
                tool_calls: None,
            },
        }],
        usage: Some(usage),
    }
}
