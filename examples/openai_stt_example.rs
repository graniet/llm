// Import required modules from the LLM library for OpenAI integration
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment variable or use test key as fallback
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // Use OpenAI as the LLM provider
        .api_key(api_key) // Set the API key
        .model("whisper-1") // Use gpt-4o-transcribe model
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .build()
        .expect("Failed to build LLM (OpenAI)");

    match llm.transcribe_file("audio2.m4a").await {
        Ok(text) => println!("Audio transcription:\n{text}"),
        Err(e) => eprintln!("Audio transcription error: {e}"),
    }

    Ok(())
}
