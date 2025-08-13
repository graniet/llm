//! Example demonstrating a multi-step chain combining speech-to-text and text processing
//!
//! This example shows how to:
//! 1. Initialize multiple LLM backends (OpenAI and ElevenLabs)
//! 2. Create a registry to manage the backends
//! 3. Build a chain that transcribes audio and processes the text
//! 4. Execute the chain and display results

use llm::{
    builder::{LLMBackend, LLMBuilder},
    chain::{LLMRegistryBuilder, MultiChainStepBuilder, MultiChainStepMode, MultiPromptChain},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize OpenAI backend with API key and model settings
    let openai_llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-OPENAI".into()))
        .model("gpt-4o")
        .build()?;

    // Initialize ElevenLabs backend for speech-to-text
    let elevenlabs_llm = LLMBuilder::new()
        .backend(LLMBackend::ElevenLabs)
        .api_key(std::env::var("ELEVENLABS_API_KEY").unwrap_or("elevenlabs-key".into()))
        .model("scribe_v1")
        .build()?;

    // Create registry to manage multiple backends
    let registry = LLMRegistryBuilder::new()
        .register("openai", openai_llm)
        .register("elevenlabs", elevenlabs_llm)
        .build();

    // Build multi-step chain using different backends
    let chain_res = MultiPromptChain::new(&registry)
        // Step 1: Transcribe audio file using ElevenLabs
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::SpeechToText)
                .provider_id("elevenlabs")
                .id("transcription")
                .template("test-stt.m4a")
                .build()?
        )
        // Step 2: Process transcription into JSON format using OpenAI
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("openai")
                .id("jsonify")
                .template("Here is the transcription: {{transcription}}\n\nPlease convert the transcription text into a JSON object with the following fields: 'text', 'words' (array of objects with 'text', 'start', 'end'), 'language_code', 'language_probability'. The JSON should be formatted as a string.")
                .build()?
        )
        .run().await?;

    // Display results from all steps
    println!("Results: {chain_res:?}");
    Ok(())
}
