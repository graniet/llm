// Example demonstrating Google Gemini's thinking/reasoning capabilities
// Gemini 2.5 and 3 series models support thinking for improved reasoning
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get Google API key from environment variable
    let api_key = std::env::var("GOOGLE_API_KEY").unwrap_or("google-key".into());

    // Example 1: Using thinking budget (Gemini 2.5 models)
    // thinking_budget controls how many tokens the model uses for reasoning
    // - Set to a positive value (128-32768 for Pro, 0-24576 for Flash) for fixed budget
    // - Set to -1 for dynamic thinking (model decides based on complexity)
    // - Set to 0 to disable thinking (for supported models)
    println!("=== Example 1: Gemini 2.5 with Thinking Budget ===\n");
    
    let llm_with_budget = LLMBuilder::new()
        .backend(LLMBackend::Google)
        .api_key(&api_key)
        .model("gemini-2.5-flash")
        .max_tokens(4096)
        .temperature(0.7)
        .reasoning_budget_tokens(1024) // Set a specific thinking budget
        .reasoning(true) // Include thought summaries in response
        .build()
        .expect("Failed to build LLM (Google)");

    let messages = vec![ChatMessage::user()
        .content("What is the sum of the first 50 prime numbers? Please show your reasoning.")
        .build()];

    match llm_with_budget.chat(&messages).await {
        Ok(response) => {
            // Check if there are thought summaries
            if let Some(thinking) = response.thinking() {
                println!("Thought Summary:\n{}\n", thinking);
            }
            if let Some(text) = response.text() {
                println!("Answer:\n{}\n", text);
            }
        }
        Err(e) => eprintln!("Chat error: {e}"),
    }

    // Example 2: Using thinking level (Gemini 3 models)
    // thinking_level controls reasoning depth:
    // - "minimal": Minimal thinking, lowest latency
    // - "low": Light reasoning, good for simple tasks
    // - "medium": Balanced (Flash only)
    // - "high": Deep reasoning, best for complex problems (default)
    println!("\n=== Example 2: Gemini 3 with Thinking Level ===\n");

    let llm_with_level = LLMBuilder::new()
        .backend(LLMBackend::Google)
        .api_key(&api_key)
        .model("gemini-3-flash-preview")
        .max_tokens(4096)
        .temperature(0.7)
        .reasoning_effort(llm::chat::ReasoningEffort::High) // Use high thinking level
        .reasoning(true) // Include thought summaries
        .build()
        .expect("Failed to build LLM (Google)");

    let complex_problem = vec![ChatMessage::user()
        .content(r#"
Alice, Bob, and Carol each live in a different house on the same street: red, green, and blue.
The person who lives in the red house owns a cat.
Bob does not live in the green house.
Carol owns a dog.
The green house is to the left of the red house.
Alice does not own a cat.
Who lives in each house, and what pet do they own?
        "#)
        .build()];

    match llm_with_level.chat(&complex_problem).await {
        Ok(response) => {
            if let Some(thinking) = response.thinking() {
                println!("Thought Summary:\n{}\n", thinking);
            }
            if let Some(text) = response.text() {
                println!("Answer:\n{}\n", text);
            }
            // Show usage information including thinking tokens
            if let Some(usage) = response.usage() {
                println!("Token Usage:");
                println!("  Prompt tokens: {}", usage.prompt_tokens);
                println!("  Completion tokens: {}", usage.completion_tokens);
                println!("  Total tokens: {}", usage.total_tokens);
            }
        }
        Err(e) => eprintln!("Chat error: {e}"),
    }

    // Example 3: Combining thinking with structured output
    println!("\n=== Example 3: Thinking with Structured Output ===\n");

    let schema = r#"
        {
            "name": "solution",
            "schema": {
                "type": "object",
                "properties": {
                    "alice_house": { "type": "string" },
                    "alice_pet": { "type": "string" },
                    "bob_house": { "type": "string" },
                    "bob_pet": { "type": "string" },
                    "carol_house": { "type": "string" },
                    "carol_pet": { "type": "string" },
                    "explanation": { "type": "string" }
                },
                "required": ["alice_house", "alice_pet", "bob_house", "bob_pet", "carol_house", "carol_pet", "explanation"]
            }
        }
    "#;
    let schema: llm::chat::StructuredOutputFormat = serde_json::from_str(schema)?;

    let llm_structured = LLMBuilder::new()
        .backend(LLMBackend::Google)
        .api_key(&api_key)
        .model("gemini-2.5-flash")
        .max_tokens(4096)
        .reasoning_budget_tokens(2048)
        .reasoning(true)
        .schema(schema)
        .build()
        .expect("Failed to build LLM (Google)");

    match llm_structured.chat(&complex_problem).await {
        Ok(response) => {
            if let Some(thinking) = response.thinking() {
                println!("Thought Summary:\n{}\n", thinking);
            }
            if let Some(text) = response.text() {
                println!("Structured Answer (JSON):\n{}\n", text);
            }
        }
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
