// Import required modules from the LLM library for xAI integration
use llm::{
    builder::{LLMBackend, LLMBuilder}, // Builder pattern components
    chat::{ChatMessage, StructuredOutputFormat}, // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get xAI API key from environment variable or use test key as fallback
    let api_key = std::env::var("XAI_API_KEY").unwrap_or("sk-TESTKEY".into());

    // Define a simple JSON schema for structured output
    let schema = r#"
        {
            "name": "student",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "age": {
                        "type": "integer"
                    },
                    "is_student": {
                        "type": "boolean"
                    }
                },
                "required": ["name", "age", "is_student"]
            }
        }
    "#;
    let schema: StructuredOutputFormat = serde_json::from_str(schema)?;

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::XAI) // Use xAI as the LLM provider
        .api_key(api_key) // Set the API key
        .model("grok-2-latest") // Use Grok-2 model
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .system("You are a helpful AI assistant. Please generate a random student using the provided JSON schema.")
        .schema(schema) // Set JSON schema for structured output
        .build()
        .expect("Failed to build LLM (xAI)");

    // Prepare conversation history with example messages
    let messages = vec![ChatMessage::user()
        .content("Please generate a random student using the provided JSON schema.")
        .build()];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(text) => println!("Chat response:\n{text}"),
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
