//! Example demonstrating XAI search -> OpenAI summary -> ElevenLabs TTS chain
//!
//! This example shows how to:
//! 1. Use XAI with search parameters to get latest Rust news
//! 2. Use OpenAI to create a bullet-point summary
//! 3. Use ElevenLabs TTS to convert the summary to speech
//! 4. Save the audio output and play it with rodio

use llm::{
    builder::{LLMBackend, LLMBuilder},
    chain::{LLMRegistryBuilder, MultiChainStepBuilder, MultiChainStepMode, MultiPromptChain},
};
use rodio::{Decoder, OutputStream, Sink};
use std::{fs::File, io::BufReader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Starting XAI Search -> OpenAI -> TTS chain example...");

    // Initialize XAI backend with search capabilities
    let xai_llm = LLMBuilder::new()
        .backend(LLMBackend::XAI)
        .api_key(std::env::var("XAI_API_KEY").unwrap_or("xai-key".into()))
        .model("grok-3-latest")
        .xai_search_mode("auto")
        .xai_max_search_results(10)
        .xai_search_from_date("2024-01-01") // Recent news
        .build()?;

    // Initialize OpenAI backend for summarization
    let openai_llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-openai".into()))
        .model("gpt-4o")
        .temperature(0.3) // Lower temperature for factual summaries
        .build()?;

    // Initialize ElevenLabs backend for text-to-speech
    let elevenlabs_llm = LLMBuilder::new()
        .backend(LLMBackend::ElevenLabs)
        .api_key(std::env::var("ELEVENLABS_API_KEY").unwrap_or("elevenlabs-key".into()))
        .model("eleven_multilingual_v2")
        .voice("JBFqnCBsd6RMkjVDRZzb")
        .build()?;

    // Create registry to manage multiple backends
    let registry = LLMRegistryBuilder::new()
        .register("xai", xai_llm)
        .register("openai", openai_llm)
        .register("elevenlabs", elevenlabs_llm)
        .build();

    println!("🔍 Step 1: Searching for latest Rust news with XAI...");

    // Build multi-step chain: XAI search -> OpenAI summary
    let chain_result = MultiPromptChain::new(&registry)
        // Step 1: Use XAI with search to get latest Rust news
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("xai")
                .id("rust_news")
                .template("Find the latest news and developments about the Rust programming language. Include recent releases, community updates, performance improvements, and notable projects. Focus on factual information from the last few months.")
                .max_tokens(1000)
                .build()?
        )
        // Step 2: Use OpenAI to create a structured bullet-point summary
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("openai")
                .id("summary")
                .template("Based on this Rust news information: {{rust_news}}\n\nCreate a clear, structured summary with the following format:\n\n• Start with 'Here are the latest Rust developments:'\n• Create 5-7 bullet points highlighting the most important news\n• Each bullet should be one concise sentence\n• Focus on releases, features, performance, and community highlights\n• End with 'That concludes the Rust news summary'\n\nMake it suitable for text-to-speech conversion.")
                .max_tokens(400)
                .temperature(0.3)
                .build()?
        )
        .run().await?;

    println!("✅ Step 2: Summary created successfully!");

    // Get the summary from chain results
    let summary = chain_result
        .get("summary")
        .ok_or("No summary found in chain results")?;

    println!("📄 Summary to be spoken:");
    println!("{summary}");
    println!();

    println!("🔊 Step 3: Converting summary to speech with ElevenLabs...");

    // Step 3: Use ElevenLabs TTS to convert summary to speech
    let elevenlabs_provider = registry
        .get("elevenlabs")
        .ok_or("ElevenLabs provider not found")?;

    let audio_data = elevenlabs_provider.speech(summary).await?;

    // Save audio to file
    let output_file = "rust_news_summary.mp3";
    std::fs::write(output_file, audio_data)?;

    println!("✅ Audio saved to: {output_file}");

    // Step 4: Play the audio using rodio
    println!("🎵 Playing audio with rodio...");

    let (_stream, stream_handle) = OutputStream::try_default()?;
    let sink = Sink::try_new(&stream_handle)?;

    let file = File::open(output_file)?;
    let source = Decoder::new(BufReader::new(file))?;

    // Play audio
    sink.append(source);

    // Block until the sound has finished
    sink.sleep_until_end();

    println!("🎉 Chain completed successfully!");
    println!();
    println!("Summary: XAI found the news -> OpenAI created bullet points -> ElevenLabs spoke it -> Rodio played it!");

    Ok(())
}
