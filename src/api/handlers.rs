#[path = "handlers/chat.rs"]
mod chat;

#[path = "handlers/chain.rs"]
mod chain;

#[path = "handlers/helpers.rs"]
mod helpers;

pub use chat::handle_chat;
