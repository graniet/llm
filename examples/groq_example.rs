// Import required modules from the LLM library for Groq integration
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Groq API key from environment variable or use test key as fallback
    let api_key = std::env::var("GROQ_API_KEY").unwrap_or("gsk-TESTKEY".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Groq) // Use Groq as the LLM provider
        .api_key(api_key) // Set the API key
        .model("deepseek-r1-distill-llama-70b") // Use deepseek-r1-distill-llama-70b model
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM (Groq)");

    // Prepare conversation history with example messages
    let messages = vec![
        ChatMessage::user()
            .content("Tell me about quantum computing")
            .build(),
        ChatMessage::assistant()
            .content("Quantum computing is a type of computing that uses quantum phenomena...")
            .build(),
        ChatMessage::user().content("What are qubits?").build(),
    ];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(text) => println!("Chat response:\n{text}"),
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
