use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use llm::error::LLMError;

use crate::runtime::AppEvent;

use super::helpers::{flush_text, flush_text_if_needed};
use super::manager::StreamRequest;

pub async fn stream_text(
    request: &StreamRequest,
    sender: &mpsc::Sender<AppEvent>,
    cancel: &CancellationToken,
) -> Result<(), LLMError> {
    let mut stream = request.provider.chat_stream(&request.messages).await?;
    let mut buffer = String::new();
    while let Some(chunk) = stream.next().await {
        if cancel.is_cancelled() {
            return Ok(());
        }
        let delta = chunk?;
        buffer.push_str(&delta);
        flush_text_if_needed(buffer.len(), &mut buffer, request, sender).await;
    }
    flush_text(&mut buffer, request, sender).await;
    Ok(())
}
