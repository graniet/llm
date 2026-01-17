#[path = "chat_wrapper/wrapper.rs"]
mod wrapper;

#[path = "chat_wrapper/impl_chat.rs"]
mod impl_chat;

#[path = "chat_wrapper/impl_other.rs"]
mod impl_other;

#[path = "chat_wrapper/reactive.rs"]
mod reactive;

pub use wrapper::{ChatWithMemory, ChatWithMemoryConfig};

#[cfg(test)]
mod tests;
