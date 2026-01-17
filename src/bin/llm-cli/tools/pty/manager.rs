//! PTY session manager for tracking active sessions.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::RwLock;

use super::super::error::ToolError;
use super::session::{PtyOutput, PtySession, SessionId};

/// Default yield time in milliseconds.
const DEFAULT_YIELD_TIME_MS: u64 = 5000;

/// Default maximum output bytes.
const DEFAULT_MAX_OUTPUT_BYTES: usize = 100_000;

/// Manages PTY sessions for the duration of a conversation.
#[derive(Debug)]
pub struct PtySessionManager {
    sessions: Arc<RwLock<HashMap<SessionId, Arc<PtySession>>>>,
    next_id: AtomicU64,
}

impl Default for PtySessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PtySessionManager {
    /// Create a new session manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicU64::new(1),
        }
    }

    /// Spawn a new PTY session with the given command.
    pub async fn spawn(
        &self,
        shell: &str,
        command: &str,
        working_dir: &str,
        yield_time_ms: Option<u64>,
    ) -> Result<PtyOutput, ToolError> {
        let id = SessionId(self.next_id.fetch_add(1, Ordering::SeqCst));
        let yield_ms = yield_time_ms.unwrap_or(DEFAULT_YIELD_TIME_MS);

        let (session, rx) = PtySession::spawn(id, shell, command, working_dir).await?;
        let session = Arc::new(session);

        // Store the session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(id, Arc::clone(&session));
        }

        // Collect initial output
        let output = session
            .collect_output(rx, yield_ms, DEFAULT_MAX_OUTPUT_BYTES)
            .await;

        // Clean up if exited
        if output.has_exited {
            self.remove(id).await;
        }

        Ok(output)
    }

    /// Write to an existing session.
    pub async fn write(
        &self,
        session_id: SessionId,
        data: &str,
        yield_time_ms: Option<u64>,
    ) -> Result<PtyOutput, ToolError> {
        let yield_ms = yield_time_ms.unwrap_or(DEFAULT_YIELD_TIME_MS);

        let session = {
            let sessions = self.sessions.read().await;
            sessions
                .get(&session_id)
                .cloned()
                .ok_or_else(|| ToolError::NotFound(format!("Session {} not found", session_id)))?
        };

        if session.has_exited() {
            self.remove(session_id).await;
            return Err(ToolError::Execution(format!(
                "Session {} has already exited",
                session_id
            )));
        }

        // Subscribe before writing
        let rx = session.subscribe();

        // Write the data
        session.write(data.as_bytes()).await?;

        // Collect output
        let output = session
            .collect_output(rx, yield_ms, DEFAULT_MAX_OUTPUT_BYTES)
            .await;

        // Clean up if exited
        if output.has_exited {
            self.remove(session_id).await;
        }

        Ok(output)
    }

    /// Get a session by ID.
    pub async fn get(&self, id: SessionId) -> Option<Arc<PtySession>> {
        let sessions = self.sessions.read().await;
        sessions.get(&id).cloned()
    }

    /// Remove a session by ID.
    pub async fn remove(&self, id: SessionId) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.remove(&id) {
            session.terminate();
        }
    }

    /// Terminate all sessions.
    pub async fn terminate_all(&self) {
        let mut sessions = self.sessions.write().await;
        for (_, session) in sessions.drain() {
            session.terminate();
        }
    }

    /// Get the number of active sessions.
    pub async fn session_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.len()
    }
}

impl Drop for PtySessionManager {
    fn drop(&mut self) {
        // Best effort cleanup - can't await in drop
        if let Ok(mut sessions) = self.sessions.try_write() {
            for (_, session) in sessions.drain() {
                session.terminate();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawn_and_collect() {
        let manager = PtySessionManager::new();
        let output = manager
            .spawn("/bin/bash", "echo hello", "/tmp", Some(1000))
            .await;

        // Should succeed on Unix
        if cfg!(unix) {
            let output = output.expect("spawn should succeed");
            assert!(output.output.contains("hello"));
        }
    }
}
