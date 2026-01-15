use std::collections::HashMap;

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use llm::chat::StreamChunk;
use llm::error::LLMError;

use crate::conversation::ToolInvocation;
use crate::runtime::{AppEvent, StreamEvent};

use super::helpers::{flush_text, flush_text_if_needed};
use super::manager::StreamRequest;

pub async fn stream_with_tools(
    request: &StreamRequest,
    sender: &mpsc::Sender<AppEvent>,
    cancel: &CancellationToken,
) -> Result<(), LLMError> {
    let tools = request.provider.tools();
    let mut stream = request
        .provider
        .chat_stream_with_tools(&request.messages, tools)
        .await?;
    let mut state = ToolStreamState::new();
    let ctx = StreamContext { request, sender };

    while let Some(chunk) = stream.next().await {
        if cancel.is_cancelled() {
            return Ok(());
        }
        let should_continue = state.handle_chunk(chunk?, &ctx).await?;
        if !should_continue {
            break;
        }
    }
    state.flush(&ctx).await;
    Ok(())
}

struct StreamContext<'a> {
    request: &'a StreamRequest,
    sender: &'a mpsc::Sender<AppEvent>,
}

struct ToolStreamState {
    buffer: String,
    tool_map: HashMap<usize, (String, String)>,
}

impl ToolStreamState {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            tool_map: HashMap::new(),
        }
    }

    async fn handle_chunk(
        &mut self,
        chunk: StreamChunk,
        ctx: &StreamContext<'_>,
    ) -> Result<bool, LLMError> {
        match chunk {
            StreamChunk::Text(delta) => {
                self.handle_text(delta, ctx).await?;
                Ok(true)
            }
            StreamChunk::ToolUseStart { index, id, name } => {
                self.handle_tool_start(index, id, name, ctx).await;
                Ok(true)
            }
            StreamChunk::ToolUseInputDelta {
                index,
                partial_json,
            } => {
                self.handle_tool_delta(index, partial_json, ctx).await;
                Ok(true)
            }
            StreamChunk::ToolUseComplete { tool_call, .. } => {
                self.handle_tool_complete(tool_call, ctx).await;
                Ok(true)
            }
            StreamChunk::Done { .. } => Ok(false),
        }
    }

    async fn handle_text(
        &mut self,
        delta: String,
        ctx: &StreamContext<'_>,
    ) -> Result<(), LLMError> {
        self.buffer.push_str(&delta);
        flush_text_if_needed(self.buffer.len(), &mut self.buffer, ctx.request, ctx.sender).await;
        Ok(())
    }

    async fn handle_tool_start(
        &mut self,
        index: usize,
        id: String,
        name: String,
        ctx: &StreamContext<'_>,
    ) {
        flush_text(&mut self.buffer, ctx.request, ctx.sender).await;
        self.tool_map.insert(index, (id.clone(), name.clone()));
        let event = StreamEvent::ToolCallStart {
            conversation_id: ctx.request.conversation_id,
            call_id: id,
            name,
        };
        let _ = ctx.sender.send(AppEvent::Stream(event)).await;
    }

    async fn handle_tool_delta(
        &mut self,
        index: usize,
        partial_json: String,
        ctx: &StreamContext<'_>,
    ) {
        flush_text(&mut self.buffer, ctx.request, ctx.sender).await;
        let Some((id, _)) = self.tool_map.get(&index) else {
            return;
        };
        let event = StreamEvent::ToolCallDelta {
            conversation_id: ctx.request.conversation_id,
            call_id: id.clone(),
            partial_json,
        };
        let _ = ctx.sender.send(AppEvent::Stream(event)).await;
    }

    async fn handle_tool_complete(&mut self, tool_call: llm::ToolCall, ctx: &StreamContext<'_>) {
        flush_text(&mut self.buffer, ctx.request, ctx.sender).await;
        let invocation = ToolInvocation::from_call(&tool_call);
        let event = StreamEvent::ToolCallComplete {
            conversation_id: ctx.request.conversation_id,
            invocation,
        };
        let _ = ctx.sender.send(AppEvent::Stream(event)).await;
    }

    async fn flush(&mut self, ctx: &StreamContext<'_>) {
        flush_text(&mut self.buffer, ctx.request, ctx.sender).await;
    }
}
