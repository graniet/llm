use std::pin::Pin;

use bytes::Bytes;
use futures::stream::{Stream, StreamExt};

use crate::error::LLMError;

const SSE_DELIMITER: &str = "\n\n";

pub(crate) fn create_sse_stream<F>(
    response: reqwest::Response,
    parser: F,
) -> Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>
where
    F: Fn(&str) -> Result<Option<String>, LLMError> + Send + 'static,
{
    let stream = response
        .bytes_stream()
        .scan(SseState::default(), move |state, chunk| {
            let results = handle_chunk(state, chunk, &parser);
            async move { Some(results) }
        })
        .flat_map(futures::stream::iter);

    Box::pin(stream)
}

#[derive(Default)]
struct SseState {
    buffer: String,
    utf8_buffer: Vec<u8>,
}

fn handle_chunk<F>(
    state: &mut SseState,
    chunk: Result<Bytes, reqwest::Error>,
    parser: &F,
) -> Vec<Result<String, LLMError>>
where
    F: Fn(&str) -> Result<Option<String>, LLMError>,
{
    let bytes = match chunk {
        Ok(bytes) => bytes,
        Err(err) => return vec![Err(LLMError::HttpError(err.to_string()))],
    };

    state.push_bytes(&bytes);
    state.drain_events(parser)
}

impl SseState {
    fn push_bytes(&mut self, bytes: &[u8]) {
        self.utf8_buffer.extend_from_slice(bytes);
        match std::str::from_utf8(&self.utf8_buffer) {
            Ok(text) => {
                self.buffer.push_str(text);
                self.utf8_buffer.clear();
            }
            Err(err) => self.consume_valid_prefix(err.valid_up_to()),
        }
    }

    fn consume_valid_prefix(&mut self, valid_up_to: usize) {
        if valid_up_to == 0 {
            return;
        }

        let valid = String::from_utf8_lossy(&self.utf8_buffer[..valid_up_to]);
        self.buffer.push_str(&valid);
        self.utf8_buffer.drain(..valid_up_to);
    }

    fn drain_events<F>(&mut self, parser: &F) -> Vec<Result<String, LLMError>>
    where
        F: Fn(&str) -> Result<Option<String>, LLMError>,
    {
        let mut results = Vec::new();
        while let Some(event) = self.next_event() {
            match parser(&event) {
                Ok(Some(content)) => results.push(Ok(content)),
                Ok(None) => {}
                Err(err) => results.push(Err(err)),
            }
        }
        results
    }

    fn next_event(&mut self) -> Option<String> {
        let pos = self.buffer.find(SSE_DELIMITER)?;
        let end = pos + SSE_DELIMITER.len();
        let event = self.buffer[..end].to_string();
        self.buffer.drain(..end);
        Some(event)
    }
}

#[cfg(test)]
#[path = "sse_tests.rs"]
mod tests;
