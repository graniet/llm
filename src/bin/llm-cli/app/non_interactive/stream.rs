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
        if let Ok(outcome) = stream_tools(handle, messages).await {
            return Ok(outcome);
        }
    }
    if let Ok(outcome) = stream_struct(handle, messages).await {
        return Ok(outcome);
    }
    if let Ok(outcome) = stream_text(handle, messages).await {
        return Ok(outcome);
    }
    chat_once(handle, messages).await
}

async fn stream_tools(
    handle: &ProviderHandle,
    messages: &[ChatMessage],
) -> Result<StreamOutcome, LLMError> {
    let tools = handle.provider.tools();
    let mut stream = handle
        .provider
        .chat_stream_with_tools(messages, tools)
        .await?;
    let mut acc = StreamAccumulator::new();
    while let Some(chunk) = stream.next().await {
        acc.apply_chunk(chunk?)?;
    }
    acc.finish()
}

async fn stream_struct(
    handle: &ProviderHandle,
    messages: &[ChatMessage],
) -> Result<StreamOutcome, LLMError> {
    let mut stream = handle.provider.chat_stream_struct(messages).await?;
    let mut acc = StreamAccumulator::new();
    while let Some(chunk) = stream.next().await {
        acc.apply_struct(chunk?)?;
    }
    acc.finish()
}

async fn stream_text(
    handle: &ProviderHandle,
    messages: &[ChatMessage],
) -> Result<StreamOutcome, LLMError> {
    let mut stream = handle.provider.chat_stream(messages).await?;
    let mut acc = StreamAccumulator::new();
    while let Some(chunk) = stream.next().await {
        acc.push_text(&chunk?)?;
    }
    acc.finish()
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
}

struct StreamPrinter {
    buffer: String,
}

impl StreamPrinter {
    fn new() -> Self {
        Self {
            buffer: String::new(),
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
