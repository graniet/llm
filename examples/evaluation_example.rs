//! Example demonstrating evaluation and comparison of multiple LLM providers
//!
//! This example shows how to:
//! 1. Initialize multiple LLM providers (Anthropic, Phind, DeepSeek)
//! 2. Configure scoring functions to evaluate responses
//! 3. Send the same prompt to all providers
//! 4. Compare and score the responses

use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
    evaluator::{EvalResult, LLMEvaluator},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Anthropic provider with Claude model
    let anthropic = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .model("claude-3-5-sonnet-20240620")
        .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or("anthropic-key".into()))
        .build()?;

    // Initialize Phind provider specialized for code
    let phind = LLMBuilder::new()
        .backend(LLMBackend::Phind)
        .model("Phind-70B")
        .build()?;

    // Initialize DeepSeek provider
    let deepseek = LLMBuilder::new()
        .backend(LLMBackend::DeepSeek)
        .model("deepseek-chat")
        .api_key(std::env::var("DEEPSEEK_API_KEY").unwrap_or("deepseek-key".into()))
        .build()?;

    // Create evaluator with multiple scoring functions
    let evaluator = LLMEvaluator::new(vec![anthropic, phind, deepseek])
        // First scoring function: Evaluate code quality and completeness
        .scoring(|response| {
            let mut score = 0.0;

            // Check for code blocks and specific Rust features
            if response.contains("```") {
                score += 1.0;

                if response.contains("```rust") {
                    score += 2.0;
                }

                if response.contains("use actix_web::") {
                    score += 2.0;
                }
                if response.contains("async fn") {
                    score += 1.0;
                }
                if response.contains("#[derive(") {
                    score += 1.0;
                }
                if response.contains("//") {
                    score += 1.0;
                }
            }

            score
        })
        // Second scoring function: Evaluate explanation quality
        .scoring(|response| {
            let mut score = 0.0;

            // Check for explanatory phrases
            if response.contains("Here's how it works:") || response.contains("Let me explain:") {
                score += 2.0;
            }

            // Check for examples and practical usage
            if response.contains("For example") || response.contains("curl") {
                score += 1.5;
            }

            // Reward comprehensive responses
            let words = response.split_whitespace().count();
            if words > 100 {
                score += 1.0;
            }

            score
        });

    // Define the evaluation prompt requesting a Rust microservice implementation
    let messages = vec![ChatMessage::user()
        .content(
            "\
            Create a Rust microservice using Actix Web.
            It should have at least two routes:
            1) A GET route returning a simple JSON status.
            2) A POST route that accepts JSON data and responds with a success message.\n\
            Include async usage, data structures with `#[derive(Serialize, Deserialize)]`, \
            and show how to run it.\n\
            Provide code blocks, comments, and a brief explanation of how it works.\
        ",
        )
        .build()];

    // Run evaluation across all providers
    let results: Vec<EvalResult> = evaluator.evaluate_chat(&messages).await?;

    // Display results with scores
    for (i, item) in results.iter().enumerate() {
        println!("\n=== LLM #{i} ===");
        println!("Score: {:.2}", item.score);
        println!("Response:\n{}", item.text);
        println!("================\n");
    }

    Ok(())
}
