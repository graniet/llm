use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use llm::error::LLMError;

use super::manager::StreamRequest;
use crate::runtime::{AppEvent, StreamEvent};

use super::stream_struct::stream_struct;
use super::stream_text::stream_text;
use super::stream_tools::stream_with_tools;

pub async fn run_stream(
    request: StreamRequest,
    sender: mpsc::Sender<AppEvent>,
    cancel: CancellationToken,
) {
    let _ = sender
        .send(AppEvent::Stream(StreamEvent::Started {
            conversation_id: request.conversation_id,
        }))
        .await;
    let start_time = Instant::now();

    if let Err(err) = stream_with_fallback(&request, &sender, &cancel).await {
        let event = StreamEvent::Error {
            conversation_id: request.conversation_id,
            message_id: request.message_id,
            error: err.to_string(),
        };
        let _ = sender.send(AppEvent::Stream(event)).await;
    }

    let _ = sender
        .send(AppEvent::Stream(StreamEvent::Done {
            conversation_id: request.conversation_id,
        }))
        .await;
    log::debug!("stream finished in {:?}", start_time.elapsed());
}

async fn stream_with_fallback(
    request: &StreamRequest,
    sender: &mpsc::Sender<AppEvent>,
    cancel: &CancellationToken,
) -> Result<(), LLMError> {
    if request.capabilities.tool_streaming
        && stream_with_tools(request, sender, cancel).await.is_ok()
    {
        return Ok(());
    }
    if stream_struct(request, sender, cancel).await.is_ok() {
        return Ok(());
    }
    stream_text(request, sender, cancel).await
}
