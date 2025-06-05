//! Reactive multi-agent example demonstrating:
//! 1. Custom broadcast capacity via `SharedMemory::new_reactive_with_capacity`.
//! 2. Cycle-limit protection with `max_cycles` to avoid infinite loops.

use llm::{
    agent::AgentBuilder,
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
    memory::{MessageCondition, SharedMemory, SlidingWindowMemory},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Shared memory with a small broadcast buffer (capacity = 50 events)
    let shared_memory =
        SharedMemory::new_reactive_with_capacity(SlidingWindowMemory::new(20), 50);

    // Proposer agent – limited to 4 reactive turns between user prompts
    let proposer = AgentBuilder::new()
        .role("assistant")
        .on_message_from_with_trigger("reviewer", MessageCondition::Contains("REJECT".to_string()))
        .max_cycles(4)
        .llm(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("gpt-3.5-turbo")
                .system("You are a proposer agent. Answer user questions. When the reviewer says REJECT, correct your answer.")
        )
        .memory(shared_memory.clone())
        .build()?;

    // Reviewer agent – replies ACCEPT / REJECT
    let _ = AgentBuilder::new()
        .role("reviewer")
        .on_message_from("assistant")
        .llm(
            LLMBuilder::new()
                .backend(LLMBackend::Anthropic)
                .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap_or("anthro-key".into()))
                .model("claude-sonnet-4-20250514")
                .system("You are a reviewer. Reply only ACCEPT or REJECT.")
        )
        .memory(shared_memory.clone())
        .build()?;

    // User task – starts the conversation
    let task = ChatMessage::user()
        .content("How many letters R are there in the word strawberry?")
        .build();
    _ = proposer.chat(&[task]).await;

    // Listen to the dialogue
    let mut rx = shared_memory.subscribe();
    while let Ok(evt) = rx.recv().await {
        println!("{}: {}", evt.role, evt.msg.content);
    }

    Ok(())
}
