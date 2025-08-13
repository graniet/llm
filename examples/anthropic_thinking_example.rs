// Import required modules from the LLM library for Anthropic integration
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Anthropic API key from environment variable or use test key as fallback
    let api_key: String = std::env::var("ANTHROPIC_API_KEY").unwrap_or("anthro-key".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic) // Use Anthropic (Claude) as the LLM provider
        .api_key(api_key) // Set the API key
        .model("claude-3-7-sonnet-20250219") // Use Claude Instant model
        .max_tokens(1500) // Limit response length
        .temperature(1.0) // Control response randomness (0.0-1.0)
        .reasoning(true)
        .reasoning_budget_tokens(1024)
        // Uncomment to set system prompt:
        // .system("You are a helpful assistant specialized in concurrency.")
        .build()
        .expect("Failed to build LLM (Anthropic)");

    // Prepare conversation history with example message about Rust concurrency
    let messages = vec![ChatMessage::user()
        .content("How much r in strawberry?")
        .build()];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(text) => {
            if let Some(thinking) = text.thinking() {
                println!("Thinking: {thinking}");
            }
            if let Some(text) = text.text() {
                println!("Text: {text}");
            }
        }
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
