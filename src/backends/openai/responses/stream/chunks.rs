use std::collections::HashMap;
use std::pin::Pin;

use bytes::Bytes;
use futures::stream::{Stream, StreamExt};

use crate::chat::StreamChunk;
use crate::error::LLMError;
use crate::{FunctionCall, ToolCall};

use super::events::{extract_payload, parse_event, ResponsesEvent, ToolState};
use super::sse::SseEventBuffer;

pub(crate) fn create_responses_stream_chunks(
    response: reqwest::Response,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>> {
    let stream = response
        .bytes_stream()
        .scan(ResponsesStreamChunkParser::new(), |parser, chunk| {
            futures::future::ready(Some(parser.handle_chunk(chunk)))
        })
        .flat_map(futures::stream::iter);
    Box::pin(stream)
}

struct ResponsesStreamChunkParser {
    sse_buffer: SseEventBuffer,
    results: Vec<Result<StreamChunk, LLMError>>,
    tool_states: HashMap<String, ToolState>,
    saw_tool_call: bool,
}

impl ResponsesStreamChunkParser {
    fn new() -> Self {
        Self {
            sse_buffer: SseEventBuffer::new(),
            results: Vec::new(),
            tool_states: HashMap::new(),
            saw_tool_call: false,
        }
    }

    fn handle_chunk(
        &mut self,
        chunk: Result<Bytes, reqwest::Error>,
    ) -> Vec<Result<StreamChunk, LLMError>> {
        match chunk {
            Ok(bytes) => self.handle_bytes(&bytes),
            Err(err) => vec![Err(LLMError::HttpError(err.to_string()))],
        }
    }

    fn handle_bytes(&mut self, bytes: &[u8]) -> Vec<Result<StreamChunk, LLMError>> {
        self.sse_buffer.push_bytes(bytes);
        for event in self.sse_buffer.drain_events() {
            self.parse_event(&event);
        }
        self.results.drain(..).collect()
    }

    fn parse_event(&mut self, event: &str) {
        let payload = match extract_payload(event) {
            Some(payload) => payload,
            None => return,
        };
        match parse_event(&payload) {
            Ok(Some(event)) => self.handle_event(event),
            Ok(None) => {}
            Err(err) => self.results.push(Err(err)),
        }
    }

    fn handle_event(&mut self, event: ResponsesEvent) {
        match event {
            ResponsesEvent::OutputTextDelta { delta } => {
                self.results.push(Ok(StreamChunk::Text(delta)));
            }
            ResponsesEvent::FunctionCallAdded {
                item_id,
                call_id,
                name,
                output_index,
            } => self.handle_call_added(item_id, call_id, name, output_index),
            ResponsesEvent::FunctionCallDelta {
                item_id,
                delta,
                output_index,
            } => self.handle_call_delta(item_id, delta, output_index),
            ResponsesEvent::FunctionCallDone {
                item_id,
                arguments,
                output_index,
            } => self.handle_call_done(item_id, arguments, output_index),
            ResponsesEvent::OutputItemDone {
                item_id,
                output_index,
            } => self.handle_item_done(item_id, output_index),
            ResponsesEvent::ResponseCompleted { .. } => {
                self.handle_response_completed();
            }
        }
    }

    fn handle_call_added(
        &mut self,
        item_id: String,
        call_id: String,
        name: String,
        output_index: usize,
    ) {
        let state = ToolState {
            call_id: call_id.clone(),
            name: name.clone(),
            arguments: String::new(),
            output_index,
        };
        self.saw_tool_call = true;
        self.results.push(Ok(StreamChunk::ToolUseStart {
            index: output_index,
            id: call_id,
            name,
        }));
        self.tool_states.insert(item_id, state);
    }

    fn handle_call_delta(&mut self, item_id: String, delta: String, output_index: usize) {
        if let Some(state) = self.tool_states.get_mut(&item_id) {
            state.arguments.push_str(&delta);
            self.results.push(Ok(StreamChunk::ToolUseInputDelta {
                index: output_index,
                partial_json: delta,
            }));
        }
    }

    fn handle_call_done(&mut self, item_id: String, arguments: String, output_index: usize) {
        if let Some(mut state) = self.tool_states.remove(&item_id) {
            if !arguments.is_empty() {
                state.arguments = arguments;
            }
            self.results.push(Ok(StreamChunk::ToolUseComplete {
                index: output_index,
                tool_call: tool_call_with_arguments(&state, &state.arguments),
            }));
        }
    }

    fn handle_item_done(&mut self, item_id: String, output_index: usize) {
        if let Some(state) = self.tool_states.remove(&item_id) {
            self.results.push(Ok(StreamChunk::ToolUseComplete {
                index: output_index,
                tool_call: tool_call_with_arguments(&state, &state.arguments),
            }));
        }
    }

    fn handle_response_completed(&mut self) {
        self.flush_tool_states();
        let stop_reason = if self.saw_tool_call {
            "tool_use"
        } else {
            "end_turn"
        };
        self.results.push(Ok(StreamChunk::Done {
            stop_reason: stop_reason.to_string(),
        }));
    }

    fn flush_tool_states(&mut self) {
        for (_, state) in self.tool_states.drain() {
            self.results.push(Ok(StreamChunk::ToolUseComplete {
                index: state.output_index,
                tool_call: tool_call_with_arguments(&state, &state.arguments),
            }));
        }
    }
}

fn tool_call_with_arguments(state: &ToolState, arguments: &str) -> ToolCall {
    ToolCall {
        id: state.call_id.clone(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: state.name.clone(),
            arguments: arguments.to_string(),
        },
    }
}

#[cfg(test)]
mod tests;
