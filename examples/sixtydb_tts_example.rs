//! Example demonstrating text-to-speech synthesis using 60db
//!
//! This example shows how to:
//! 1. Initialize the 60db text-to-speech provider
//! 2. Generate speech from text
//! 3. Save the audio output to a file

use llm::builder::{LLMBackend, LLMBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment variable or use test key
    let api_key = std::env::var("SIXTYDB_API_KEY").unwrap_or("test_key".into());

    // Initialize 60db text-to-speech provider
    let tts = LLMBuilder::new()
        .backend(LLMBackend::SixtyDb)
        .api_key(api_key)
        // Optional: set a voice id from your 60db account (GET /myvoices).
        // .voice("fbb75ed2-975a-40c7-9e06-38e30524a9a1")
        .build()?;

    // Text to convert to speech
    let text = "Hello! This is an example of text-to-speech synthesis using 60db.";

    // Generate speech
    let audio_data = tts.speech(text).await?;

    // Save the audio to a file
    std::fs::write("output-speech-60db.mp3", audio_data)?;

    println!("Audio file generated successfully: output-speech-60db.mp3");
    Ok(())
}
