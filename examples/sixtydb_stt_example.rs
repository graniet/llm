//! Example demonstrating speech-to-text transcription using 60db
//!
//! This example shows how to:
//! 1. Initialize the 60db speech-to-text provider
//! 2. Load an audio file
//! 3. Transcribe the audio content

use llm::builder::{LLMBackend, LLMBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment variable or use test key
    let api_key = std::env::var("SIXTYDB_API_KEY").unwrap_or("test_key".into());

    // Initialize 60db speech-to-text provider
    let stt = LLMBuilder::new()
        .backend(LLMBackend::SixtyDb)
        .api_key(api_key)
        .build()?;

    // Read audio file from disk (WAV, MP3, M4A, OGG, FLAC, WebM, MP4)
    let audio_bytes = std::fs::read("test-stt.m4a")?;

    // Transcribe audio content
    let resp = stt.transcribe(audio_bytes).await?;

    println!("Transcription: {resp}");
    Ok(())
}
