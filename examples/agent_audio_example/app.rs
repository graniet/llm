use anyhow::{Context, Result};
use llm::{
    agent::AgentBuilder,
    builder::{LLMBackend, LLMBuilder},
    cond,
    memory::{SharedMemory, SlidingWindowMemory},
    LLMProvider,
};
use std::sync::Arc;
use tokio::runtime::Runtime;

const MEMORY_WINDOW: usize = 10;
const REQUEST_TIMEOUT_SECS: u64 = 30;

const TRANSCRIBER_MODEL: &str = "gpt-4o";
const TRANSCRIBER_STT_MODEL: &str = "whisper-1";
const ASSISTANT_MODEL: &str = "gpt-4o-search-preview";

const TRANSCRIBER_SYSTEM: &str = "You are a planner. Use only the transcription. If it is empty \
or unclear, reply with NO_TRANSCRIPT and nothing else. Otherwise provide a short plan as bullet \
points. Do not ask questions.";
const ASSISTANT_SYSTEM: &str = "Execute the plan and respond to the user.";

pub(crate) struct AppContext {
    pub(crate) memory: SharedMemory<SlidingWindowMemory>,
    pub(crate) transcriber: Arc<dyn LLMProvider>,
    _assistant: Arc<dyn LLMProvider>,
}

pub fn run() -> Result<()> {
    let runtime = build_runtime()?;
    let handle = runtime.handle().clone();
    let ctx = runtime.block_on(setup_agents())?;
    crate::runtime::run(ctx, handle)
}

fn build_runtime() -> Result<Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build runtime")
}

async fn setup_agents() -> Result<AppContext> {
    let api_key = std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY not set")?;
    let memory = SharedMemory::new_reactive(SlidingWindowMemory::new(MEMORY_WINDOW));

    let transcriber = AgentBuilder::new()
        .role("transcriber")
        .llm(openai_builder(&api_key, TRANSCRIBER_MODEL).system(TRANSCRIBER_SYSTEM))
        .stt(openai_builder(&api_key, TRANSCRIBER_STT_MODEL))
        .memory(memory.clone())
        .build()
        .context("failed to build transcriber")?;

    let assistant = AgentBuilder::new()
        .role("assistant")
        .on("transcriber", cond!(not_contains "NO_TRANSCRIPT"))
        .llm(
            openai_builder(&api_key, ASSISTANT_MODEL)
                .system(ASSISTANT_SYSTEM)
                .openai_enable_web_search(true),
        )
        .memory(memory.clone())
        .build()
        .context("failed to build assistant")?;

    Ok(AppContext {
        memory,
        transcriber: Arc::from(transcriber),
        _assistant: Arc::from(assistant),
    })
}

fn openai_builder(api_key: &str, model: &str) -> LLMBuilder {
    LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .model(model)
        .timeout_seconds(REQUEST_TIMEOUT_SECS)
}
