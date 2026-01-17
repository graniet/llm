use std::{collections::HashMap, pin::Pin};

use bytes::Bytes;
use futures::stream::{Stream, StreamExt};

use crate::{
    chat::{StreamResponse, Usage},
    error::LLMError,
};

use super::events::{extract_payload, parse_event, ResponsesEvent as Event, ToolState};
use super::response_helpers::{
    stream_response_text, stream_response_tool_call, stream_response_usage, tool_call_from_state,
    tool_call_with_arguments,
};
use super::sse::SseEventBuffer;

pub(crate) fn create_responses_stream_responses(
    response: reqwest::Response,
    normalize_response: bool,
) -> Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>> {
    let stream = response
        .bytes_stream()
        .scan(
            ResponsesStreamResponseParser::new(normalize_response),
            |parser, chunk| futures::future::ready(Some(parser.handle_chunk(chunk))),
        )
        .flat_map(futures::stream::iter);
    Box::pin(stream)
}

struct ResponsesStreamResponseParser {
    sse_buffer: SseEventBuffer,
    results: Vec<Result<StreamResponse, LLMError>>,
    tool_states: HashMap<String, ToolState>,
    normalize_response: bool,
}

impl ResponsesStreamResponseParser {
    fn new(normalize_response: bool) -> Self {
        Self {
            sse_buffer: SseEventBuffer::new(),
            results: Vec::new(),
            tool_states: HashMap::new(),
            normalize_response,
        }
    }

    fn handle_chunk(
        &mut self,
        chunk: Result<Bytes, reqwest::Error>,
    ) -> Vec<Result<StreamResponse, LLMError>> {
        match chunk {
            Ok(bytes) => self.handle_bytes(&bytes),
            Err(err) => vec![Err(LLMError::HttpError(err.to_string()))],
        }
    }

    fn handle_bytes(&mut self, bytes: &[u8]) -> Vec<Result<StreamResponse, LLMError>> {
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

    fn handle_event(&mut self, event: Event) {
        match event {
            Event::OutputTextDelta { delta } => self.handle_output_text(delta),
            Event::FunctionCallAdded {
                item_id,
                call_id,
                name,
                output_index,
            } => self.handle_function_call_added(item_id, call_id, name, output_index),
            Event::FunctionCallDelta {
                item_id,
                delta,
                output_index,
            } => self.handle_function_call_delta(item_id, delta, output_index),
            Event::FunctionCallDone {
                item_id,
                arguments,
                output_index,
            } => self.handle_function_call_done(item_id, arguments, output_index),
            Event::OutputItemDone { item_id, .. } => self.handle_output_item_done(&item_id),
            Event::ResponseCompleted { usage } => self.handle_response_completed(usage),
        }
    }

    fn handle_output_text(&mut self, delta: String) {
        self.results.push(Ok(stream_response_text(delta)));
    }

    fn handle_function_call_added(
        &mut self,
        item_id: String,
        call_id: String,
        name: String,
        output_index: usize,
    ) {
        let state = ToolState {
            call_id,
            name,
            arguments: String::new(),
            output_index,
        };
        if !self.normalize_response {
            self.results
                .push(Ok(stream_response_tool_call(tool_call_from_state(&state))));
        }
        self.tool_states.insert(item_id, state);
    }

    fn handle_function_call_delta(&mut self, item_id: String, delta: String, _output_index: usize) {
        if let Some(state) = self.tool_states.get_mut(&item_id) {
            state.arguments.push_str(&delta);
            if !self.normalize_response {
                self.results
                    .push(Ok(stream_response_tool_call(tool_call_with_arguments(
                        state, &delta,
                    ))));
            }
        }
    }

    fn handle_function_call_done(
        &mut self,
        item_id: String,
        arguments: String,
        _output_index: usize,
    ) {
        if let Some(mut state) = self.tool_states.remove(&item_id) {
            if !arguments.is_empty() {
                state.arguments = arguments;
            }
            if self.normalize_response {
                self.results
                    .push(Ok(stream_response_tool_call(tool_call_with_arguments(
                        &state,
                        &state.arguments,
                    ))));
            }
        }
    }

    fn handle_output_item_done(&mut self, item_id: &str) {
        if self.normalize_response {
            self.finish_tool_call(item_id);
        } else {
            self.tool_states.remove(item_id);
        }
    }

    fn handle_response_completed(&mut self, usage: Option<Usage>) {
        if self.normalize_response {
            self.finish_all_tool_calls();
        } else {
            self.tool_states.clear();
        }
        if let Some(usage) = usage {
            self.results.push(Ok(stream_response_usage(usage)));
        }
    }

    fn finish_tool_call(&mut self, item_id: &str) {
        if let Some(state) = self.tool_states.remove(item_id) {
            self.results
                .push(Ok(stream_response_tool_call(tool_call_with_arguments(
                    &state,
                    &state.arguments,
                ))));
        }
    }

    fn finish_all_tool_calls(&mut self) {
        for (_, state) in self.tool_states.drain() {
            self.results
                .push(Ok(stream_response_tool_call(tool_call_with_arguments(
                    &state,
                    &state.arguments,
                ))));
        }
    }
}

#[cfg(test)]
mod tests;
