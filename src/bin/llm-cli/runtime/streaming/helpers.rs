use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;

use super::manager::StreamRequest;
use crate::runtime::{AppEvent, StreamEvent};

pub const TOKEN_BATCH_SIZE: usize = 32;

/// Wraps the event sender so `stream_with_fallback` can tell, after a
/// streaming attempt fails partway through, whether any event already
/// reached the UI. A fallback attempt must never retry with a different
/// streaming method once that has happened - the user would see the failed
/// attempt's partial output immediately followed by an unrelated, independent
/// response.
pub struct EmittingSender<'a> {
    inner: &'a mpsc::Sender<AppEvent>,
    emitted: AtomicBool,
}

impl<'a> EmittingSender<'a> {
    pub fn new(inner: &'a mpsc::Sender<AppEvent>) -> Self {
        Self {
            inner,
            emitted: AtomicBool::new(false),
        }
    }

    pub async fn send(&self, event: AppEvent) -> Result<(), mpsc::error::SendError<AppEvent>> {
        self.emitted.store(true, Ordering::Relaxed);
        self.inner.send(event).await
    }

    pub fn emitted(&self) -> bool {
        self.emitted.load(Ordering::Relaxed)
    }
}

pub async fn flush_text_if_needed(
    size: usize,
    buffer: &mut String,
    request: &StreamRequest,
    sender: &EmittingSender<'_>,
) {
    if size >= TOKEN_BATCH_SIZE {
        flush_text(buffer, request, sender).await;
    }
}

pub async fn flush_text(buffer: &mut String, request: &StreamRequest, sender: &EmittingSender<'_>) {
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
