// Import required modules from the LLM library
use futures::StreamExt;
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Ollama server URL from environment variable or use default localhost
    let base_url = std::env::var("OLLAMA_URL").unwrap_or("http://127.0.0.1:11434".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Ollama) // Use Ollama as the LLM backend
        .base_url(base_url) // Set the Ollama server URL
        .model("llama3.2:latest")
        .max_tokens(1000) // Set maximum response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(true) // Enable streaming responses
        .build()
        .expect("Failed to build LLM (Ollama)");

    // Prepare conversation history with example messages
    let messages = vec![ChatMessage::user()
        .content("Write a short poem about Rust programming language")
        .build()];

    // Send streaming chat request and handle the response
    match llm.chat_stream(&messages).await {
        Ok(mut stream) => {
            println!("Ollama streaming chat response:");
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(text) => print!("{}", text),
                    Err(e) => eprintln!("Stream error: {}", e),
                }
            }
            println!(); // Add newline at the end
        }
        Err(e) => eprintln!("Chat stream error: {}", e),
    }

    Ok(())
}
