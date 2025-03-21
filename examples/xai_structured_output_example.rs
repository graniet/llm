use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, StructuredOutputFormat},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("XAI_API_KEY").unwrap_or("sk-TESTKEY".into());

    let schema = r#"
    {
        "name": "Student",
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

    let llm = LLMBuilder::new()
        .backend(LLMBackend::XAI)
        .api_key(api_key)
        .model("grok-2-latest")
        .max_tokens(512)
        .temperature(0.7)
        .stream(false)
        .system("You are a helpful AI assistant. Please generate a random student using the provided JSON schema.")
        .schema(schema)
        .build()
        .expect("Failed to build LLM (xAI)");

    let messages = vec![ChatMessage::user()
        .content("Please generate a random student using the provided JSON schema.")
        .build()];

    match llm.chat(&messages).await {
        Ok(text) => println!("Chat response:\n{}", text),
        Err(e) => eprintln!("Chat error: {}", e),
    }

    Ok(())
}
