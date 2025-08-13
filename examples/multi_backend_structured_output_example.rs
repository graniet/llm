//! Send the same JSON schema to multiple backends for structured output.

use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, StructuredOutputFormat},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into());

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

    let messages = vec![ChatMessage::user()
        .content("Generate a random student")
        .build()];

    let llm_openai = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .model("gpt-4o")
        .max_tokens(512)
        .temperature(0.7)
        .stream(false)
        .system("You are an AI assistant that can provide structured output to generate random students as example data. Respond in JSON format using the provided JSON schema.")
        .schema(schema.clone())
        .build()
        .expect("Failed to build LLM (OpenAI)");

    match llm_openai.chat(&messages).await {
        Ok(text) => println!("Chat response:\n{text}"),
        Err(e) => eprintln!("Chat error: {e}"),
    }

    let llm_ollama = LLMBuilder::new()
        .backend(LLMBackend::Ollama)
        .base_url("http://127.0.0.1:11434")
        .model("llama3.1:latest")
        .max_tokens(512)
        .temperature(0.7)
        .stream(false)
        .schema(schema)
        .system("You are a helpful AI assistant. Please generate a random student using the provided JSON schema.")
        .build()
        .expect("Failed to build LLM (Ollama)");

    match llm_ollama.chat(&messages).await {
        Ok(text) => println!("Ollama chat response:\n{text}"),
        Err(e) => eprintln!("Chat error: {e}"),
    }

    Ok(())
}
