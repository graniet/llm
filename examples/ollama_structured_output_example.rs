// Import required modules from the LLM library
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, StructuredOutputFormat},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Ollama server URL from environment variable or use default localhost
    let base_url = std::env::var("OLLAMA_URL").unwrap_or("http://127.0.0.1:11434".into());

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
        .backend(LLMBackend::Ollama) // Use Ollama as the LLM backend
        .base_url(base_url) // Set the Ollama server URL
        .model("llama3.1:latest")
        .max_tokens(1000) // Set maximum response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .schema(schema) // Set JSON schema for structured output
        .system("You are a helpful AI assistant. Please generate a random student using the provided JSON schema.")
        .build()
        .expect("Failed to build LLM (Ollama)");

    // Prepare conversation history with example messages
    let messages = vec![ChatMessage::user()
        .content("Please generate a random student using the provided JSON schema.")
        .build()];

    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(text) => println!("Ollama chat response:\n{text}"),
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
