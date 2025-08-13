// Memory sharing example
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
    memory::{SharedMemory, SlidingWindowMemory},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shared_memory = SharedMemory::new(SlidingWindowMemory::new(10));

    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("gpt-3.5-turbo")
        .memory(shared_memory.clone())
        .build()?;

    let llm2 = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or("anthro-key".into()))
        .model("claude-3-7-sonnet-20250219")
        .max_tokens(512)
        .temperature(0.7)
        .memory(shared_memory)
        .build()?;

    let messages1 = vec![ChatMessage::user().content("My name is Alice").build()];
    match llm.chat(&messages1).await {
        Ok(response) => println!("Response 1: {response}"),
        Err(e) => eprintln!("Error 1: {e}"),
    }

    let messages2 = vec![ChatMessage::user().content("What's my name?").build()];
    match llm2.chat(&messages2).await {
        Ok(response) => println!("Response 2: {response}"),
        Err(e) => eprintln!("Error 2: {e}"),
    }

    Ok(())
}
