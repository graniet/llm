// Import required modules from the LLM library for OpenAI integration
use llm::{
    builder::{LLMBackend, LLMBuilder},    // Builder pattern components
    chat::{ChatMessage, ReasoningEffort}, // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment variable or use test key as fallback
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // Use OpenAI as the LLM provider
        .api_key(api_key) // Set the API key
        .model("o1-preview") // Use GPT-3.5 Turbo model
        .stream(false) // Disable streaming responses
        .reasoning_effort(ReasoningEffort::High) // Enable reasoning effort
        .build()
        .expect("Failed to build LLM (OpenAI)");

    // Prepare conversation history with example messages
    let messages = vec![ChatMessage::user()
        .content("How muck r in strawberry")
        .build()];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(text) => println!("Chat response:\n{text}"),
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
