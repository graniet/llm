// Import required modules from the LLM library for Ollama integration
use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::ChatMessage, // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::Ollama) // Use Ollama as the LLM provider
        .base_url("http://localhost:11434") // Local Ollama server
        .model("llama3") // Use Llama3 model or whatever is available on your local server
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .function(
            FunctionBuilder::new("weather_function")
                .description("Use this tool to get the weather in a specific city")
                .param(
                    ParamBuilder::new("city")
                        .type_of("string")
                        .description("The city to get the weather for"),
                )
                .required(vec!["city".to_string()]),
        )
        .build()
        .expect("Failed to build LLM");

    // Prepare conversation history with example messages
    let messages = vec![ChatMessage::user().content("You are a weather assistant. What is the weather in Tokyo? Use the tools that you have available").build()];

    // Send chat request and handle the response
    match llm.chat_with_tools(&messages, llm.tools()).await {
        Ok(text) => println!("Chat response:\n{}", text),
        Err(e) => eprintln!("Chat error: {}", e),
    }

    Ok(())
}