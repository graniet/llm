//! Voice assistant example with audio input and a two-agent pipeline.
//!
//! Controls:
//! - SPACE: start/stop recording
//! - q: quit
//!
//! Required environment variables:
//! - OPENAI_API_KEY

#[path = "agent_audio_example/app.rs"]
mod app;
#[path = "agent_audio_example/audio.rs"]
mod audio;
#[path = "agent_audio_example/runtime.rs"]
mod runtime;
#[path = "agent_audio_example/ui.rs"]
mod ui;

fn main() -> anyhow::Result<()> {
    app::run()
}
