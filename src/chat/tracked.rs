//! Tracked stream wrapper for collecting metrics during streaming.
//!
//! This module provides [`Tracked`], a generic stream wrapper that collects
//! timing and usage metrics as chunks are consumed.
//!
//! # Example
//!
//! ```rust,ignore
//! use llm::chat::Tracked;
//! use futures::StreamExt;
//!
//! let stream = provider.chat_stream_with_tools(messages, None).await?;
//! let mut tracked = Tracked::new(stream);
//!
//! while let Some(chunk) = tracked.next().await {
//!     match chunk? {
//!         StreamChunk::Text(text) => print!("{}", text),
//!         StreamChunk::Done { .. } => break,
//!         _ => {}
//!     }
//! }
//!
//! let metrics = tracked.finalize();
//! println!("Time to first token: {:?}", metrics.time_to_first_token);
//! println!("Total duration: {:?}", metrics.duration);
//! println!("Tokens/sec: {:?}", metrics.tokens_per_second());
//! ```

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;

use futures::Stream;
use pin_project::pin_project;

use crate::error::LLMError;
use crate::ToolCall;

use super::{ChatMetrics, StreamChunk, StreamResponse, Usage};

/// Trait for stream items that can be tracked for metrics.
///
/// Implement this trait for custom stream item types to enable
/// tracking with [`Tracked`].
pub trait Trackable {
    /// Extract text content from this item, if any.
    fn extract_text(&self) -> Option<&str>;

    /// Extract a completed tool call from this item, if any.
    fn extract_tool_call(&self) -> Option<&ToolCall>;

    /// Extract usage statistics from this item, if any.
    fn extract_usage(&self) -> Option<&Usage>;

    /// Returns true if this item indicates the stream is done.
    fn is_done(&self) -> bool;
}

impl Trackable for StreamChunk {
    fn extract_text(&self) -> Option<&str> {
        match self {
            StreamChunk::Text(t) => Some(t),
            _ => None,
        }
    }

    fn extract_tool_call(&self) -> Option<&ToolCall> {
        match self {
            StreamChunk::ToolUseComplete { tool_call, .. } => Some(tool_call),
            _ => None,
        }
    }

    fn extract_usage(&self) -> Option<&Usage> {
        None // StreamChunk doesn't carry usage
    }

    fn is_done(&self) -> bool {
        matches!(self, StreamChunk::Done { .. })
    }
}

impl Trackable for StreamResponse {
    fn extract_text(&self) -> Option<&str> {
        self.choices.first()?.delta.content.as_deref()
    }

    fn extract_tool_call(&self) -> Option<&ToolCall> {
        None // Tool calls come through delta.tool_calls but aren't complete
    }

    fn extract_usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    fn is_done(&self) -> bool {
        false // StreamResponse doesn't have explicit done marker
    }
}

impl Trackable for String {
    fn extract_text(&self) -> Option<&str> {
        Some(self)
    }

    fn extract_tool_call(&self) -> Option<&ToolCall> {
        None
    }

    fn extract_usage(&self) -> Option<&Usage> {
        None
    }

    fn is_done(&self) -> bool {
        false
    }
}

/// A stream wrapper that tracks metrics as chunks are consumed.
///
/// `Tracked` wraps any stream and collects timing information and
/// accumulated content as items are polled. After draining the stream,
/// call [`finalize()`](Tracked::finalize) to get the collected metrics.
///
/// # Type Parameters
///
/// * `S` - The inner stream type
///
/// # Example
///
/// ```rust,ignore
/// let stream = provider.chat_stream_with_tools(messages, None).await?;
/// let mut tracked = Tracked::new(stream);
///
/// while let Some(chunk) = tracked.next().await {
///     // Process chunk...
/// }
///
/// let metrics = tracked.finalize();
/// println!("Duration: {:?}", metrics.duration);
/// ```
#[pin_project]
pub struct Tracked<S> {
    #[pin]
    inner: S,
    start_time: Instant,
    first_chunk_time: Option<Instant>,
    accumulated_text: String,
    tool_calls: Vec<ToolCall>,
    usage: Option<Usage>,
    chunk_count: usize,
}

impl<S> Tracked<S> {
    /// Create a new tracked stream wrapper.
    ///
    /// The timer starts immediately when this is called.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            start_time: Instant::now(),
            first_chunk_time: None,
            accumulated_text: String::new(),
            tool_calls: Vec::new(),
            usage: None,
            chunk_count: 0,
        }
    }

    /// Finalize and get metrics.
    ///
    /// Can be called at any time, but metrics are most meaningful
    /// after the stream has been fully drained.
    pub fn finalize(&self) -> ChatMetrics {
        ChatMetrics {
            usage: self.usage.clone(),
            duration: self.start_time.elapsed(),
            time_to_first_token: self
                .first_chunk_time
                .map(|t| t.duration_since(self.start_time)),
        }
    }

    /// Get the accumulated text so far.
    pub fn text(&self) -> &str {
        &self.accumulated_text
    }

    /// Get the tool calls collected so far.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    /// Get the number of chunks received so far.
    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    /// Get the usage statistics if available.
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }
}

impl<S, T> Stream for Tracked<S>
where
    S: Stream<Item = Result<T, LLMError>>,
    T: Trackable,
{
    type Item = Result<T, LLMError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        match this.inner.poll_next(cx) {
            Poll::Ready(Some(Ok(item))) => {
                *this.chunk_count += 1;

                // Extract and accumulate
                if let Some(text) = item.extract_text() {
                    // Record time to first token only on actual text content
                    if !text.is_empty() && this.first_chunk_time.is_none() {
                        *this.first_chunk_time = Some(Instant::now());
                    }
                    this.accumulated_text.push_str(text);
                }
                if let Some(tool_call) = item.extract_tool_call() {
                    this.tool_calls.push(tool_call.clone());
                }
                if let Some(usage) = item.extract_usage() {
                    *this.usage = Some(usage.clone());
                }

                Poll::Ready(Some(Ok(item)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_tracked_accumulates_text() {
        let chunks = vec![
            Ok(StreamChunk::Text("Hello ".to_string())),
            Ok(StreamChunk::Text("world".to_string())),
            Ok(StreamChunk::Done {
                stop_reason: "end_turn".to_string(),
            }),
        ];
        let stream = stream::iter(chunks);
        let mut tracked = Tracked::new(stream);

        while let Some(_) = tracked.next().await {}

        assert_eq!(tracked.text(), "Hello world");
        assert_eq!(tracked.chunk_count(), 3);
    }

    #[tokio::test]
    async fn test_tracked_records_first_chunk_time() {
        let chunks = vec![
            Ok(StreamChunk::Text("Hi".to_string())),
            Ok(StreamChunk::Done {
                stop_reason: "end_turn".to_string(),
            }),
        ];
        let stream = stream::iter(chunks);
        let mut tracked = Tracked::new(stream);

        // Before any chunks
        assert!(tracked.first_chunk_time.is_none());

        // Consume first chunk
        let _ = tracked.next().await;
        assert!(tracked.first_chunk_time.is_some());

        let metrics = tracked.finalize();
        assert!(metrics.time_to_first_token.is_some());
    }

    #[tokio::test]
    async fn test_tracked_collects_tool_calls() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: crate::FunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location": "Paris"}"#.to_string(),
            },
        };

        let chunks = vec![
            Ok(StreamChunk::ToolUseStart {
                index: 0,
                id: "call_123".to_string(),
                name: "get_weather".to_string(),
            }),
            Ok(StreamChunk::ToolUseComplete {
                index: 0,
                tool_call: tool_call.clone(),
            }),
            Ok(StreamChunk::Done {
                stop_reason: "tool_use".to_string(),
            }),
        ];
        let stream = stream::iter(chunks);
        let mut tracked = Tracked::new(stream);

        while let Some(_) = tracked.next().await {}

        assert_eq!(tracked.tool_calls().len(), 1);
        assert_eq!(tracked.tool_calls()[0].function.name, "get_weather");
    }

    #[tokio::test]
    async fn test_tracked_finalize_returns_metrics() {
        let chunks = vec![
            Ok(StreamChunk::Text("Test".to_string())),
            Ok(StreamChunk::Done {
                stop_reason: "end_turn".to_string(),
            }),
        ];
        let stream = stream::iter(chunks);
        let mut tracked = Tracked::new(stream);

        while let Some(_) = tracked.next().await {}

        let metrics = tracked.finalize();
        assert!(metrics.duration.as_nanos() > 0);
        assert!(metrics.time_to_first_token.is_some());
        // Usage is None since StreamChunk doesn't carry it
        assert!(metrics.usage.is_none());
    }

    #[tokio::test]
    async fn test_tracked_with_string_stream() {
        let chunks: Vec<Result<String, LLMError>> = vec![
            Ok("Hello ".to_string()),
            Ok("world".to_string()),
        ];
        let stream = stream::iter(chunks);
        let mut tracked = Tracked::new(stream);

        while let Some(_) = tracked.next().await {}

        assert_eq!(tracked.text(), "Hello world");
        assert_eq!(tracked.chunk_count(), 2);
    }

    #[test]
    fn test_trackable_stream_chunk_text() {
        let chunk = StreamChunk::Text("hello".to_string());
        assert_eq!(chunk.extract_text(), Some("hello"));
        assert!(chunk.extract_tool_call().is_none());
        assert!(!chunk.is_done());
    }

    #[test]
    fn test_trackable_stream_chunk_done() {
        let chunk = StreamChunk::Done {
            stop_reason: "end_turn".to_string(),
        };
        assert!(chunk.is_done());
        assert!(chunk.extract_text().is_none());
    }

    #[test]
    fn test_trackable_string() {
        let s = "test".to_string();
        assert_eq!(s.extract_text(), Some("test"));
        assert!(!s.is_done());
    }
}
