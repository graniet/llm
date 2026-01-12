use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use llm::chat::StreamResponse;
use llm::error::LLMError;

use crate::conversation::ToolInvocation;
use crate::runtime::{AppEvent, StreamEvent};

use super::helpers::{flush_text, flush_text_if_needed};
use super::manager::StreamRequest;

pub async fn stream_struct(
    request: &StreamRequest,
    sender: &mpsc::Sender<AppEvent>,
    cancel: &CancellationToken,
) -> Result<(), LLMError> {
    let mut stream = request
        .provider
        .chat_stream_struct(&request.messages)
        .await?;
    let mut buffer = String::new();
    while let Some(chunk) = stream.next().await {
        if cancel.is_cancelled() {
            return Ok(());
        }
        handle_chunk(chunk?, request, sender, &mut buffer).await;
    }
    flush_text(&mut buffer, request, sender).await;
    Ok(())
}

async fn handle_chunk(
    chunk: StreamResponse,
    request: &StreamRequest,
    sender: &mpsc::Sender<AppEvent>,
    buffer: &mut String,
) {
    handle_usage(&chunk, request, sender).await;
    handle_delta(&chunk, request, sender, buffer).await;
}

async fn handle_usage(
    chunk: &StreamResponse,
    request: &StreamRequest,
    sender: &mpsc::Sender<AppEvent>,
) {
    let Some(usage) = chunk.usage.clone() else {
        return;
    };
    let event = StreamEvent::Usage {
        conversation_id: request.conversation_id,
        message_id: request.message_id,
        usage,
    };
    let _ = sender.send(AppEvent::Stream(event)).await;
}

async fn handle_delta(
    chunk: &StreamResponse,
    request: &StreamRequest,
    sender: &mpsc::Sender<AppEvent>,
    buffer: &mut String,
) {
    let Some(choice) = chunk.choices.first() else {
        return;
    };
    if let Some(content) = &choice.delta.content {
        buffer.push_str(content);
        flush_text_if_needed(buffer.len(), buffer, request, sender).await;
    }
    if let Some(tool_calls) = &choice.delta.tool_calls {
        for call in tool_calls {
            let invocation = ToolInvocation::from_call(call);
            let event = StreamEvent::ToolCallComplete {
                conversation_id: request.conversation_id,
                invocation,
            };
            let _ = sender.send(AppEvent::Stream(event)).await;
        }
    }
}
