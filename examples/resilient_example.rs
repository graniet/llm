//! Example demonstrating the ResilientLLM wrapper with retry/backoff.
//!
//! Run with:
//! `cargo run --example resilient_example --features openai`

use llm::builder::{LLMBackend, LLMBuilder};
use llm::chat::ChatMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    llm::init_logging();

    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .model("gpt-4o-mini")
        .resilient(true)
        .resilient_attempts(3)
        .resilient_backoff(200, 2000)
        .build()?;

    let messages = vec![ChatMessage::user()
        .content("Reply with a single short greeting.")
        .build()];

    let response = llm.chat(&messages).await?;
    println!("{}", response);
    Ok(())
}


