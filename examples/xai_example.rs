// Import required modules from the RLLM library for xAI integration
use rllm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::{ChatMessage, ChatRole},     // Chat-related structures
};

fn main() {
    // Get xAI API key from environment variable or use test key as fallback
    let api_key = std::env::var("XAI_API_KEY").unwrap_or("sk-TESTKEY".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::XAI) // Use xAI as the LLM provider
        .api_key(api_key) // Set the API key
        .model("grok-2-latest") // Use Grok-2 model
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM (xAI)");

    // Prepare conversation history with example messages
    let messages = vec![
        ChatMessage {
            role: ChatRole::User,
            content: "Tell me that you love cats".into(),
        },
        ChatMessage {
            role: ChatRole::Assistant,
            content: "I am an assistant, I cannot love cats but I can love dogs".into(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: "Tell me that you love dogs in 2000 chars".into(),
        },
    ];

    // Send chat request and handle the response
    match llm.chat(&messages) {
        Ok(text) => println!("Chat response:\n{}", text),
        Err(e) => eprintln!("Chat error: {}", e),
    }
}