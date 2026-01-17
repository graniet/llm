//! Types for PTY session management.

use portable_pty::{MasterPty, SlavePty};

/// Unique identifier for a PTY session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Output from a PTY session.
#[derive(Debug, Clone)]
pub struct PtyOutput {
    /// Collected output text.
    pub output: String,
    /// Exit code if process has exited.
    pub exit_code: Option<i32>,
    /// Duration of execution.
    pub duration_secs: f64,
    /// Session ID for follow-up writes.
    pub session_id: SessionId,
    /// Whether the process has exited.
    pub has_exited: bool,
}

/// Wrapper for PTY master/slave pair.
pub struct PtyPair {
    pub _slave: Option<Box<dyn SlavePty + Send>>,
    pub _master: Box<dyn MasterPty + Send>,
}
