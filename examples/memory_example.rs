// Memory integration example
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create LLM with automatic memory
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("gpt-3.5-turbo")
        .sliding_window_memory(5)
        .build()?;

    // First conversation
    let messages1 = vec![ChatMessage::user().content("My name is Alice").build()];
    match llm.chat(&messages1).await {
        Ok(response) => println!("Response 1: {response}"),
        Err(e) => eprintln!("Error 1: {e}"),
    }

    // Second conversation - should remember Alice's name
    let messages2 = vec![ChatMessage::user().content("What's my name?").build()];
    match llm.chat(&messages2).await {
        Ok(response) => println!("Response 2: {response}"),
        Err(e) => eprintln!("Error 2: {e}"),
    }

    Ok(())
}
