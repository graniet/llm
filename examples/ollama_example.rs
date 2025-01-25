// Import required modules from the LLM library
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, ChatRole},
};

fn main() {
    // Get Ollama server URL from environment variable or use default localhost
    let base_url = std::env::var("OLLAMA_URL").unwrap_or("http://localhost:11434".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Ollama) // Use Ollama as the LLM backend
        .base_url(base_url) // Set the Ollama server URL
        .model("llama3.2") // Use the Mistral model
        .max_tokens(1000) // Set maximum response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM (Ollama)");

    // Prepare conversation history with example messages
    let messages = vec![
        ChatMessage {
            role: ChatRole::User,
            content: "Hello, how do I run a local LLM in Rust?".into(),
        },
        ChatMessage {
            role: ChatRole::Assistant,
            content: "One way is to use Ollama with a local model!".into(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: "Tell me more about that".into(),
        },
    ];

    // Send chat request and handle the response
    match llm.chat(&messages) {
        Ok(text) => println!("Ollama chat response:\n{}", text),
        Err(e) => eprintln!("Chat error: {}", e),
    }
}
