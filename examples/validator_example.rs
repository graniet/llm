// Import required modules from the LLM library
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder components for LLM configuration
    chat::ChatMessage,                 // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Retrieve Anthropic API key from environment variable or use fallback
    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or("anthro-key".into());

    // Initialize and configure the LLM client with validation
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic) // Use Anthropic's Claude model
        .model("claude-3-5-sonnet-20240620") // Specify model version
        .api_key(api_key) // Set API credentials
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness
        .stream(false) // Disable streaming responses
        .validator(|resp| {
            // Add JSON validation
            serde_json::from_str::<serde_json::Value>(resp)
                .map(|_| ())
                .map_err(|e| e.to_string())
        })
        .validator_attempts(3) // Allow up to 3 retries on validation failure
        .build()
        .expect("Failed to build LLM (Phind)");

    // Prepare the chat message requesting JSON output
    let messages = vec![
        ChatMessage::user().content("Please give me a valid JSON describing a cat named Garfield, color 'orange'. with format {name: string, color: string}. Return only the JSON, no other text").build(),
    ];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(text) => println!("{text}"),
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
