//! Port of `reactive_agent_example.rs` to the new `AgentBuilder`.
//!
//! Three agents cooperate via a shared reactive memory:
//! 1. proposer (assistant) - answers user questions.
//! 2. reviewer - judges the answer (ACCEPT / REJECT).
//! 3. resumer - summarizes the discussion when the answer is accepted.

use llm::{
    agent::AgentBuilder,
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
    cond,
    memory::{SharedMemory, SlidingWindowMemory},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shared_memory = SharedMemory::new_reactive(SlidingWindowMemory::new(10));

    let proposer = AgentBuilder::new()
        .role("assistant")
        .on("user", cond!(any))
        .on("reviewer", cond!(contains "REJECT"))
        .llm(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("gpt-3.5-turbo")
                .system("You are a proposer agent. Answer user questions accurately and concisely. When you receive REJECT from a reviewer, correct your previous response.")
        )
        .stt(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("whisper-1")
        )
        .tts(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("tts-1")
                .voice("alloy")
        )
        .memory(shared_memory.clone())
        .build()?;

    let _reviewer = AgentBuilder::new()
        .role("reviewer")
        .on("assistant", cond!(any))
        .debounce(800)
        .llm(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("o3")
                .system("Respond with a single word: ACCEPT (if the assistant is correct) or REJECT (if wrong). No explanation..")
                .validator(|resp| {
                    if resp == "ACCEPT" || resp == "REJECT" {
                        Ok(())
                    } else {
                        Err("Invalid response".to_string())
                    }
                })
                .validator_attempts(3)
        )
        .memory(shared_memory.clone())
        .build()?;

    let _resumer = AgentBuilder::new()
        .role("resumer")
        .on("reviewer", cond!(contains "ACCEPT"))
        .llm(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into()))
                .model("gpt-4o")
                .system("You are a resumer agent. Summarize the conversation between the proposer and the reviewer."),
        )
        .memory(shared_memory.clone())
        .build()?;

    let task = ChatMessage::user()
        .content("how much R in the word strawberry ?")
        .build();
    _ = proposer.chat(&[task]).await;

    let mut receiver = shared_memory.subscribe();
    while let Ok(evt) = receiver.recv().await {
        println!("{} said: {}", evt.role, evt.msg.content);
    }

    Ok(())
}
