use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
    memory::TrimStrategy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
        .model("gpt-3.5-turbo")
        .sliding_window_with_strategy(3, TrimStrategy::Summarize)
        .build()?;

    println!("Testing TrimStrategy::Summarize with window size 3");

    let messages = vec![
        ChatMessage::user()
            .content("Hello, my name is Alice")
            .build(),
        ChatMessage::user()
            .content("I love programming in Rust")
            .build(),
        ChatMessage::user()
            .content("What's the weather like?")
            .build(),
        ChatMessage::user().content("Tell me about AI").build(),
    ];

    for (i, message) in messages.iter().enumerate() {
        println!("\n--- Message {} ---", i + 1);
        println!("Sending: {}", message.content);

        match llm.chat(&[message.clone()]).await {
            Ok(response) => println!("Response: {response}"),
            Err(e) => eprintln!("Error: {e}"),
        }

        if let Some(memory_contents) = llm.memory_contents().await {
            println!("\nMemory contents ({} messages):", memory_contents.len());
            for (j, msg) in memory_contents.iter().enumerate() {
                let role = match msg.role {
                    llm::chat::ChatRole::User => "User",
                    llm::chat::ChatRole::Assistant => "Assistant",
                };
                println!("  {}: {} - '{}'", j + 1, role, msg.content);
            }
        }
    }

    Ok(())
}
