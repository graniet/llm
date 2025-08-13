// Import required modules from the LLM library for OpenAI integration
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::ChatMessage,                 // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment variable or use test key as fallback
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI) // Use OpenAI as the LLM provider
        .api_key(api_key) // Set the API key
        .model("gpt-4o-search-preview") // Use gpt-4o-search-preview Turbo model
        .max_tokens(512) // Limit response length
        .stream(false) // Disable streaming responses
        .openai_enable_web_search(true) // Enable web search
        .openai_web_search_context_size("low") // Set search context
        .openai_web_search_user_location_type("approximate") // Set search context type
        .openai_web_search_user_location_approximate_country("US") // Set search context country
        .openai_web_search_user_location_approximate_city("Los Angeles") // Set search context city
        .openai_web_search_user_location_approximate_region("California") // Set search context region
        .build()
        .expect("Failed to build LLM (OpenAI)");

    // Prepare conversation history with example messages
    let messages = vec![ChatMessage::user()
        .content("What are the latest news from Los Angeles?")
        .build()];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(text) => println!("Chat response:\n{text}"),
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
