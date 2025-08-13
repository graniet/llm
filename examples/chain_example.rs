//! Example demonstrating a multi-step prompt chain for exploring programming language features
//!
//! This example shows how to:
//! 1. Select a programming language topic
//! 2. Get advanced features for that language
//! 3. Generate a code example for one feature
//! 4. Get a detailed explanation of the example

use llm::{
    builder::{LLMBackend, LLMBuilder},
    chain::{ChainStepBuilder, ChainStepMode, PromptChain},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the LLM with OpenAI backend and configuration
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("gpt-4o")
        .max_tokens(200)
        .temperature(0.7)
        .build()?;

    // Create and execute a 4-step prompt chain
    let chain_result = PromptChain::new(&*llm)
        // Step 1: Choose a programming language topic
        .step(
            ChainStepBuilder::new("topic", "Suggest an interesting technical topic to explore among: Rust, Python, JavaScript, Go. Answer with a single word only.", ChainStepMode::Chat)
                .temperature(0.8) // Higher temperature for more variety in topic selection
                .build()
        )
        // Step 2: Get advanced features for the chosen language
        .step(
            ChainStepBuilder::new("features", "List 3 advanced features of {{topic}} that few developers know about. Format: one feature per line.", ChainStepMode::Chat)
                .build()
        )
        // Step 3: Generate a code example for one feature
        .step(
            ChainStepBuilder::new("example", "Choose one of the features listed in {{features}} and show a commented code example that illustrates it.", ChainStepMode::Chat)
                .build()
        )
        // Step 4: Get detailed explanation of the code example
        .step(
            ChainStepBuilder::new("explanation", "Explain in detail how the code example {{example}} works and why this feature is useful.", ChainStepMode::Chat)
                .max_tokens(500) // Allow longer response for detailed explanation
                .build()
        )
        .run().await?;

    // Display the results from all chain steps
    println!("Chain results: {chain_result:?}");

    Ok(())
}
