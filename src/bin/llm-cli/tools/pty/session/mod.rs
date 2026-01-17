//! PTY session implementation using portable-pty.

#![allow(dead_code)]

mod spawn;
pub mod types;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use tokio::sync::{broadcast, mpsc};
use tokio::task::JoinHandle;

pub use spawn::spawn_session;
pub use types::{PtyOutput, PtyPair, SessionId};

/// A PTY session that manages an interactive shell process.
pub struct PtySession {
    pub(crate) id: SessionId,
    pub(crate) writer_tx: mpsc::Sender<Vec<u8>>,
    pub(crate) output_tx: broadcast::Sender<Vec<u8>>,
    pub(crate) exit_status: Arc<AtomicBool>,
    pub(crate) exit_code: Arc<StdMutex<Option<i32>>>,
    pub(crate) reader_handle: StdMutex<Option<JoinHandle<()>>>,
    pub(crate) writer_handle: StdMutex<Option<JoinHandle<()>>>,
    pub(crate) wait_handle: StdMutex<Option<JoinHandle<()>>>,
    pub(crate) killer: StdMutex<Option<Box<dyn portable_pty::ChildKiller + Send + Sync>>>,
    pub(crate) _pair: StdMutex<PtyPair>,
}

impl std::fmt::Debug for PtySession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PtySession")
            .field("id", &self.id)
            .field("has_exited", &self.has_exited())
            .finish()
    }
}

impl PtySession {
    /// Spawn a new PTY session with the given command.
    pub async fn spawn(
        id: SessionId,
        shell: &str,
        command: &str,
        working_dir: &str,
    ) -> Result<(Self, broadcast::Receiver<Vec<u8>>), super::super::error::ToolError> {
        spawn_session(id, shell, command, working_dir).await
    }

    /// Create default exit state atomics.
    pub(crate) fn default_exit_state() -> (Arc<AtomicBool>, Arc<StdMutex<Option<i32>>>) {
        (
            Arc::new(AtomicBool::new(false)),
            Arc::new(StdMutex::new(None)),
        )
    }

    /// Get the session ID.
    pub const fn id(&self) -> SessionId {
        self.id
    }

    /// Check if the process has exited.
    pub fn has_exited(&self) -> bool {
        self.exit_status.load(Ordering::SeqCst)
    }

    /// Get the exit code if the process has exited.
    pub fn exit_code(&self) -> Option<i32> {
        self.exit_code.lock().ok().and_then(|g| *g)
    }

    /// Write bytes to the PTY stdin.
    pub async fn write(&self, data: &[u8]) -> Result<(), super::super::error::ToolError> {
        use super::super::error::ToolError;
        self.writer_tx
            .send(data.to_vec())
            .await
            .map_err(|_| ToolError::Execution("Failed to write to PTY".to_string()))
    }

    /// Subscribe to output stream.
    pub fn subscribe(&self) -> broadcast::Receiver<Vec<u8>> {
        self.output_tx.subscribe()
    }

    /// Collect output for a duration, then return.
    pub async fn collect_output(
        &self,
        mut rx: broadcast::Receiver<Vec<u8>>,
        yield_time_ms: u64,
        max_output_bytes: usize,
    ) -> PtyOutput {
        let start = Instant::now();
        let yield_duration = Duration::from_millis(yield_time_ms);
        let mut output = Vec::new();

        loop {
            let remaining = yield_duration.saturating_sub(start.elapsed());
            if remaining.is_zero() || output.len() >= max_output_bytes {
                break;
            }

            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Ok(chunk)) => {
                    output.extend_from_slice(&chunk);
                }
                Ok(Err(_)) => break,
                Err(_) => break,
            }
        }

        let output_str = String::from_utf8_lossy(&output).to_string();
        let duration_secs = start.elapsed().as_secs_f64();

        PtyOutput {
            output: output_str,
            exit_code: self.exit_code(),
            duration_secs,
            session_id: self.id,
            has_exited: self.has_exited(),
        }
    }

    /// Terminate the session.
    pub fn terminate(&self) {
        if let Ok(mut killer_opt) = self.killer.lock() {
            if let Some(mut killer) = killer_opt.take() {
                let _ = killer.kill();
            }
        }

        for handle in [&self.reader_handle, &self.writer_handle, &self.wait_handle] {
            if let Ok(mut h) = handle.lock() {
                if let Some(handle) = h.take() {
                    handle.abort();
                }
            }
        }
    }
}

impl Drop for PtySession {
    fn drop(&mut self) {
        self.terminate();
    }
}
