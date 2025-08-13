use std::{
    fs,
    io::{self, Write},
};

use futures::StreamExt;
// Import required modules from the LLM library for Google Gemini integration
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::{ChatMessage, ImageMime},    // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Ollama) // Use Google as the LLM provider
        .model("qwen2.5vl:7b") // Use Gemini Pro model
        .max_tokens(8512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        // Optional: Set system prompt
        .system("You are a helpful AI assistant specialized in programming.")
        .build()
        .expect("Failed to build LLM (Ollama)");

    let content = fs::read("./examples/image001.jpg").expect("The image001.jpg file should exist");

    // Prepare conversation history with example messages
    let messages = vec![
        ChatMessage::user()
            .content("Explain what you see in the image")
            .build(),
        ChatMessage::user().image(ImageMime::JPEG, content).build(),
    ];

    // Send chat request and handle the response
    match llm.chat_stream(&messages).await {
        Ok(mut stream) => {
            while let Some(delta) = stream.next().await {
                print!("{}", delta.unwrap_or("".to_owned()));
                io::stdout().flush().expect("failed to flush");
            }
            println!() //Print a newline
        }
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
