use tokio::sync::mpsc;

use super::manager::StreamRequest;
use crate::runtime::{AppEvent, StreamEvent};

pub const TOKEN_BATCH_SIZE: usize = 32;

pub async fn flush_text_if_needed(
    size: usize,
    buffer: &mut String,
    request: &StreamRequest,
    sender: &mpsc::Sender<AppEvent>,
) {
    if size >= TOKEN_BATCH_SIZE {
        flush_text(buffer, request, sender).await;
    }
}

pub async fn flush_text(
    buffer: &mut String,
    request: &StreamRequest,
    sender: &mpsc::Sender<AppEvent>,
) {
    if buffer.is_empty() {
        return;
    }
    let delta = std::mem::take(buffer);
    let event = StreamEvent::TextDelta {
        conversation_id: request.conversation_id,
        message_id: request.message_id,
        delta,
    };
    let _ = sender.send(AppEvent::Stream(event)).await;
}
