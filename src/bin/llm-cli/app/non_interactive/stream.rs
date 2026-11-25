use std::io::{self, Write};

use futures::StreamExt;
use llm::chat::{ChatMessage, StreamChunk, StreamResponse};
use llm::error::LLMError;
use llm::ToolCall;

use crate::provider::ProviderHandle;

const STREAM_FLUSH_THRESHOLD: usize = 32;

pub(super) struct StreamOutcome {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
}

pub(super) async fn stream_once(
    handle: &ProviderHandle,
    messages: &[ChatMessage],
) -> Result<StreamOutcome, LLMError> {
    if handle.capabilities.tool_streaming {
        match stream_tools(handle, messages).await {
            (Ok(outcome), _) => return Ok(outcome),
            (Err(err), true) => return Err(err),
            (Err(_), false) => {}
        }
    }
    match stream_struct(handle, messages).await {
        (Ok(outcome), _) => return Ok(outcome),
        (Err(err), true) => return Err(err),
        (Err(_), false) => {}
    }
    match stream_text(handle, messages).await {
        (Ok(outcome), _) => return Ok(outcome),
        (Err(err), true) => return Err(err),
        (Err(_), false) => {}
    }
    chat_once(handle, messages).await
}

/// Runs a streaming attempt to completion, returning whether any output was
/// already printed even when it fails partway through. A fallback attempt
/// must never be retried once real output has been printed for a prior one -
/// see `stream_once` - or the user sees the failed attempt's partial output
/// immediately followed by an unrelated, independent response.
async fn stream_tools(
    handle: &ProviderHandle,
    messages: &[ChatMessage],
) -> (Result<StreamOutcome, LLMError>, bool) {
    let tools = handle.provider.tools();
    let mut stream = match handle
        .provider
        .chat_stream_with_tools(messages, tools)
        .await
    {
        Ok(stream) => stream,
        Err(err) => return (Err(err), false),
    };
    let mut acc = StreamAccumulator::new();
    while let Some(chunk) = stream.next().await {
        let result = match chunk {
            Ok(chunk) => acc.apply_chunk(chunk),
            Err(err) => Err(err),
        };
        if let Err(err) = result {
            return (Err(err), acc.printed_any());
        }
    }
    let printed_any = acc.printed_any();
    (acc.finish(), printed_any)
}

async fn stream_struct(
    handle: &ProviderHandle,
    messages: &[ChatMessage],
) -> (Result<StreamOutcome, LLMError>, bool) {
    let mut stream = match handle.provider.chat_stream_struct(messages).await {
        Ok(stream) => stream,
        Err(err) => return (Err(err), false),
    };
    let mut acc = StreamAccumulator::new();
    while let Some(chunk) = stream.next().await {
        let result = match chunk {
            Ok(chunk) => acc.apply_struct(chunk),
            Err(err) => Err(err),
        };
        if let Err(err) = result {
            return (Err(err), acc.printed_any());
        }
    }
    let printed_any = acc.printed_any();
    (acc.finish(), printed_any)
}

async fn stream_text(
    handle: &ProviderHandle,
    messages: &[ChatMessage],
) -> (Result<StreamOutcome, LLMError>, bool) {
    let mut stream = match handle.provider.chat_stream(messages).await {
        Ok(stream) => stream,
        Err(err) => return (Err(err), false),
    };
    let mut acc = StreamAccumulator::new();
    while let Some(chunk) = stream.next().await {
        let result = match chunk {
            Ok(delta) => acc.push_text(&delta),
            Err(err) => Err(err),
        };
        if let Err(err) = result {
            return (Err(err), acc.printed_any());
        }
    }
    let printed_any = acc.printed_any();
    (acc.finish(), printed_any)
}

async fn chat_once(
    handle: &ProviderHandle,
    messages: &[ChatMessage],
) -> Result<StreamOutcome, LLMError> {
    let response = handle
        .provider
        .chat_with_tools(messages, handle.provider.tools())
        .await?;
    let text = response.text().unwrap_or_default();
    let tool_calls = response.tool_calls().unwrap_or_default();
    print_blocking(&text)?;
    Ok(StreamOutcome { text, tool_calls })
}

struct StreamAccumulator {
    text: String,
    tool_calls: Vec<ToolCall>,
    printer: StreamPrinter,
}

impl StreamAccumulator {
    fn new() -> Self {
        Self {
            text: String::new(),
            tool_calls: Vec::new(),
            printer: StreamPrinter::new(),
        }
    }

    fn push_text(&mut self, delta: &str) -> Result<(), LLMError> {
        self.text.push_str(delta);
        self.printer.push(delta)?;
        Ok(())
    }

    fn apply_chunk(&mut self, chunk: StreamChunk) -> Result<(), LLMError> {
        match chunk {
            StreamChunk::Text(delta) => self.push_text(&delta),
            StreamChunk::ToolUseComplete { tool_call, .. } => {
                self.tool_calls.push(tool_call);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn apply_struct(&mut self, chunk: StreamResponse) -> Result<(), LLMError> {
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                self.push_text(content)?;
            }
            if let Some(tool_calls) = &choice.delta.tool_calls {
                self.tool_calls.extend(tool_calls.iter().cloned());
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<StreamOutcome, LLMError> {
        self.printer.flush()?;
        Ok(StreamOutcome {
            text: self.text,
            tool_calls: self.tool_calls,
        })
    }

    /// Whether any content has already reached stdout for this attempt.
    /// Once true, a caller must not retry with a different streaming method
    /// on failure - see `stream_once`.
    fn printed_any(&self) -> bool {
        self.printer.printed_any
    }
}

struct StreamPrinter {
    buffer: String,
    printed_any: bool,
}

impl StreamPrinter {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            printed_any: false,
        }
    }

    fn push(&mut self, delta: &str) -> Result<(), LLMError> {
        self.buffer.push_str(delta);
        if self.buffer.len() >= STREAM_FLUSH_THRESHOLD {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), LLMError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        print_blocking(&self.buffer)?;
        self.printed_any = true;
        self.buffer.clear();
        Ok(())
    }
}

fn print_blocking(text: &str) -> Result<(), LLMError> {
    let mut stdout = io::stdout();
    stdout
        .write_all(text.as_bytes())
        .and_then(|_| stdout.flush())
        .map_err(|err| LLMError::Generic(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printed_any_stays_false_until_content_actually_reaches_stdout() {
        let mut acc = StreamAccumulator::new();
        assert!(!acc.printed_any());
        // Below the flush threshold: buffered internally, nothing printed yet.
        acc.push_text("short").unwrap();
        assert!(!acc.printed_any());
    }

    #[test]
    fn printed_any_becomes_true_once_the_flush_threshold_is_crossed() {
        // This is exactly the signal stream_once's fallback logic checks
        // when a streaming attempt errors partway through: once true, a
        // fallback to a different streaming method must not be attempted,
        // since the user has already seen this attempt's partial output.
        let mut acc = StreamAccumulator::new();
        acc.push_text(&"x".repeat(STREAM_FLUSH_THRESHOLD)).unwrap();
        assert!(acc.printed_any());
    }
}
