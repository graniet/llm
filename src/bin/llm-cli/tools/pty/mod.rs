//! PTY (Pseudo-Terminal) session management for interactive shell execution.
//!
//! Provides persistent shell sessions with stdin/stdout streaming.

#![allow(unused_imports)]

mod manager;
pub mod session;

pub use manager::PtySessionManager;
pub use session::{PtyOutput, PtySession, SessionId};
