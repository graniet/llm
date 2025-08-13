//! Example demonstrating how to chain multiple LLM backends together
//!
//! This example shows how to:
//! 1. Initialize multiple LLM backends (OpenAI, Anthropic)
//! 2. Create a registry to manage multiple backends
//! 3. Build a multi-step chain that uses different backends at each step
//! 4. Pass results between steps using template variables

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

    // Initialize Anthropic backend with API key and model settings
    let anthro_llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or("anthro-key".into()))
        .model("claude-3-5-sonnet-20240620")
        .build()?;

    let deepseek_llm = LLMBuilder::new()
        .backend(LLMBackend::DeepSeek)
        .api_key(std::env::var("DEEPSEEK_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("deepseek-chat")
        .build()?;

    // Ollama backend could also be added
    // let ollama_llm = LLMBuilder::new()
    //     .backend(LLMBackend::Ollama)
    //     .base_url("http://localhost:11411")
    //     .model("mistral")
    //     .build()?;

    // Create registry to manage multiple backends
    let registry = LLMRegistryBuilder::new()
        .register("openai", openai_llm)
        .register("anthro", anthro_llm)
        .register("deepseek", deepseek_llm)
        // .register("ollama", ollama_llm)
        .build();

    // Build multi-step chain using different backends
    let chain_res = MultiPromptChain::new(&registry)
        // Step 1: Use OpenAI to analyze a code problem
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("openai")
                .id("analysis")
                .template("Analyze this Rust code and identify potential performance issues:\n```rust\nfn process_data(data: Vec<i32>) -> Vec<i32> {\n    data.iter().map(|x| x * 2).collect()\n}```")
                .temperature(0.7)
                .build()?
        )
        // Step 2: Use Anthropic to suggest optimizations based on analysis
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("anthro")
                .id("optimization")
                .template("Here is a code analysis: {{analysis}}\n\nSuggest concrete optimizations to improve performance, explaining why they would be beneficial.")
                .max_tokens(500)
                .top_p(0.9)
                .build()?
        )
        // Step 3: Use OpenAI to generate optimized code
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("openai")
                .id("final_code")
                .template("Taking into account these optimization suggestions: {{optimization}}\n\nGenerate an optimized version of the code in Rust with explanatory comments.")
                .temperature(0.2)
                .build()?
        )
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("deepseek")
                .id("final_code")
                .template("Taking into account these optimization suggestions: {{optimization}}\n\nGenerate an optimized version of the code in Rust with explanatory comments.")
                .temperature(0.2)
                .build()?
        )
        .run().await?;

    // Display results from all steps
    println!("Results: {chain_res:?}");
    // Example output format:
    // chain_res["analysis"] => "The code has potential performance issues..."
    // chain_res["optimization"] => "Here are some suggested optimizations..."
    // chain_res["final_code"] => "// Optimized version with comments..."

    Ok(())
}
