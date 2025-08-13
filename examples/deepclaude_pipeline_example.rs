//! Example demonstrating a multi-LLM pipeline for system identification
//!
//! This example shows how to:
//! 1. Set up a pipeline using Groq and Claude models
//! 2. Use Groq for creative system identification approaches
//! 3. Use Claude to convert ideas into concrete commands
//! 4. Transform and filter responses between steps
//! 5. Handle results in a type-safe way with error handling

use llm::{
    builder::{LLMBackend, LLMBuilder},
    chain::{LLMRegistryBuilder, MultiChainStepBuilder, MultiChainStepMode, MultiPromptChain},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Claude model with API key and latest model version
    let anthro_llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or("anthro-key".into()))
        .model("claude-3-5-sonnet-20240620")
        .build()?;

    // Initialize Groq model with deepseek for creative thinking
    let groq_llm = LLMBuilder::new()
        .backend(LLMBackend::Groq)
        .api_key(std::env::var("GROQ_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("deepseek-r1-distill-llama-70b")
        .build()?;

    // Create registry with both models
    let registry = LLMRegistryBuilder::new()
        .register("anthro", anthro_llm)
        .register("groq", groq_llm)
        .build();

    // Build and execute the multi-step chain
    let chain_res = MultiPromptChain::new(&registry)
        // Step 1: Use Groq to generate creative system identification approaches
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("groq")
                .id("thinking")
                .template("Find an original way to identify the system without using default commands. I want a one-line command.")
                .max_tokens(500)
                .top_p(0.9)
                // Transform response to extract only content between <think> tags
                .response_transform(|resp| {
                    resp.lines()
                        .skip_while(|line| !line.contains("<think>"))
                        .take_while(|line| !line.contains("</think>"))
                        .map(|line| line.replace("<think>", "").trim().to_string())
                        .filter(|line| !line.is_empty())
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .build()?
        )
        // Step 2: Use Claude to convert the creative approach into a concrete command
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("anthro")
                .id("command")
                .template("Take the following command reasoning and generate a command to execute it on the system: {{thinking}}\n\nGenerate a command to execute it on the system. return only the command.")
                .temperature(0.2) // Low temperature for more deterministic output
                .build()?
        )
        .run().await?;

    // Display results from both steps
    println!("Results: {chain_res:?}");

    Ok(())
}
