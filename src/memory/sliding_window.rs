#[path = "sliding_window/core.rs"]
mod core;

#[path = "sliding_window/provider.rs"]
mod provider;

pub use core::{SlidingWindowMemory, TrimStrategy, WindowSize};
