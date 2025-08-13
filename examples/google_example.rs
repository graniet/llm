// Import required modules from the LLM library for Google Gemini integration
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Google API key from environment variable or use test key as fallback
    let api_key = std::env::var("GOOGLE_API_KEY").unwrap_or("google-key".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Google) // Use Google as the LLM provider
        .api_key(api_key) // Set the API key
        .model("gemini-2.0-flash-exp") // Use Gemini Pro model
        .max_tokens(8512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        // Optional: Set system prompt
        .system("You are a helpful AI assistant specialized in programming.")
        .build()
        .expect("Failed to build LLM (Google)");

    // Prepare conversation history with example messages
    let messages = vec![
        ChatMessage::user()
            .content("Explain the concept of async/await in Rust")
            .build(),
        ChatMessage::assistant()
            .content("Async/await in Rust is a way to write asynchronous code...")
            .build(),
        ChatMessage::user()
            .content("Can you show me a simple example?")
            .build(),
    ];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(text) => println!("Google Gemini response:\n{text}"),
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
