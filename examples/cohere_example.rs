// Import necessary modules for Cohere backend
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder components
    chat::ChatMessage,                 // Chat message structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Cohere API key from environment variable (or use a dummy key as default)
    let api_key = std::env::var("COHERE_API_KEY").unwrap_or("test-key-123".into());

    // Initialize and configure the LLM client with Cohere
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Cohere) // Use Cohere as LLM provider
        .api_key(api_key) // Set API key
        .model("command-a-03-2025") // Choose a Cohere model (free-tier)
        .system("Answer like a pirate.") // System instruction (sent with 'developer' role)
        .max_tokens(200) // Limit response length
        .temperature(0.7) // Set response creativity
        .build()
        .expect("Failed to build LLM (Cohere)");

    // Prepare conversation history with a user message
    let messages = vec![ChatMessage::user().content("What is 2 + 2?").build()];

    // Send chat request and display response or error
    match llm.chat(&messages).await {
        Ok(response) => println!("Cohere model response:\n{response}"),
        Err(e) => eprintln!("Error calling Cohere: {e}"),
    }

    Ok(())
}
