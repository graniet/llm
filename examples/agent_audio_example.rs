//! A voice assistant example that demonstrates how to use the LLM library with audio input.
//!
//! This example creates a terminal UI application that:
//! - Records audio from the default input device when spacebar is pressed
//! - Transcribes the audio using OpenAI's Whisper model
//! - Processes the transcribed text using a two-agent pipeline:
//!   1. A transcriber agent that creates a plan based on the audio input
//!   2. An assistant agent that executes the plan and responds to the user
//!
//! The UI shows:
//! - A scrollable list of messages from the agent pipeline
//! - Recording controls and status at the bottom
//!
//! # Usage
//! - Press SPACE to start/stop recording
//! - Press 'q' to quit
//!
//! # Required Environment Variables
//! - OPENAI_API_KEY: Your OpenAI API key

use llm::{
    agent::AgentBuilder,
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
    cond,
    memory::{SharedMemory, SlidingWindowMemory},
};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame, Terminal,
};

use std::{
    collections::VecDeque,
    io::{self, Cursor},
    sync::{Arc, Mutex},
};

use tokio::sync::mpsc;

/// Messages that can be sent to update the UI
#[derive(Debug, Clone)]
enum AppMessage {
    /// A message from an agent, including role, content and whether it contains audio
    Agent { role: String, content: String, audio: bool },
    /// A status message to display in the UI
    Status(String),
}

/// The main application state
struct App {
    /// Queue of messages to display in the UI
    messages: VecDeque<String>,
    /// Current status message
    status: String,
    /// Whether audio is currently being recorded
    recording: bool,
}

impl App {
    /// Maximum number of messages to keep in history
    const MAX: usize = 512;

    /// Creates a new application instance
    fn new() -> Self {
        Self {
            messages: VecDeque::from(vec!["🚀 Voice Assistant Agent Started".into()]),
            status: "Press SPACE to record, 'q' to quit".into(),
            recording: false,
        }
    }

    /// Adds a new message to the history
    fn push(&mut self, role: &str, content: &str, audio: bool) {
        if self.messages.len() == Self::MAX {
            self.messages.pop_front();
        }
        let icon = if audio { "🎵" } else { "📝" };
        let msg = format!("{icon} [{role}]: {content}");
        self.messages.push_back(msg);
    }

    /// Updates the status message
    fn set_status(&mut self, s: impl Into<String>) {
        self.status = s.into();
    }

    /// Updates the recording state and status message
    fn set_recording(&mut self, rec: bool) {
        self.recording = rec;
        self.set_status(if rec {
            "🔴 Recording… Press SPACE to stop"
        } else {
            "Press SPACE to record, 'q' to quit"
        });
    }
}

/// Draws the terminal UI
fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)])
        .split(f.area());

    let items: Vec<_> = app
        .messages
        .iter()
        .map(|m| ListItem::new(m.clone()))
        .collect();
    f.render_widget(
        List::new(items).block(
            Block::default()
                .title("🤖 Agent Pipeline Messages")
                .borders(Borders::ALL),
        ),
        chunks[0],
    );

    let style = if app.recording {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Green)
    };
    f.render_widget(
        Paragraph::new(app.status.clone())
            .block(
                Block::default()
                    .title("🎙️ Recording Controls")
                    .borders(Borders::ALL),
            )
            .style(style),
        chunks[1],
    );
}

/// Main application entry point
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let memory = SharedMemory::new_reactive(SlidingWindowMemory::new(10));

    let transcriber = Arc::new(
        AgentBuilder::new()
            .role("transcriber")
            .on("user", cond!(has_audio))
            .llm(
                LLMBuilder::new()
                    .backend(LLMBackend::OpenAI)
                    .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TEST".into()))
                    .model("gpt-4o")
                    .system("Create a plan to answer the question. You are not allowed to change the content of the audio message. You are not allowed to add any other text to the transcribed text."),
            )
            .stt(
                LLMBuilder::new()
                    .backend(LLMBackend::OpenAI)
                    .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TEST".into()))
                    .model("whisper-1"),
            )
            .memory(memory.clone())
            .build()?,
    );

    let _assistant = AgentBuilder::new()
        .role("assistant")
        .on("transcriber", cond!(any))
        .llm(
            LLMBuilder::new()
                .backend(LLMBackend::OpenAI)
                .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("sk-TEST".into()))
                .model("gpt-4o-search-preview")
                .openai_enable_web_search(true)
                .system("Execute the plan and respond to the user."),
        )
        .memory(memory.clone())
        .build()?;

    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    let host = cpal::default_host();
    let input_dev = host
        .default_input_device()
        .expect("no input device available");
    let cfg = input_dev.default_input_config()?;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let (ui_tx, mut ui_rx) = mpsc::channel::<AppMessage>(128);

    {
        let ui_tx = ui_tx.clone();
        tokio::spawn(async move {
            let mut sub = memory.subscribe();
            while let Ok(e) = sub.recv().await {
                let audio = e.msg.has_audio();
                let content = e.msg.content;
                let _ = ui_tx
                    .send(AppMessage::Agent {
                        role: e.role,
                        content,
                        audio,
                    })
                    .await;
            }
        });
    }

    let samples = Arc::new(Mutex::new(Vec::<f32>::with_capacity(48_000 * 60)));

    let mut stream: Option<cpal::Stream> = None;

    loop {
        while let Ok(msg) = ui_rx.try_recv() {
            match msg {
                AppMessage::Agent { role, content, audio } => app.push(&role, &content, audio),
                AppMessage::Status(s) => app.set_status(s),
            }
        }

        terminal.draw(|f| draw(f, &app))?;

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(k) = event::read()? {
                match k.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char(' ') => {
                        if app.recording {
                            if let Some(s) = stream.take() {
                                drop(s);
                            }
                            app.set_recording(false);
                            app.set_status("🎵 Processing audio…");

                            let pcm: Vec<f32> = {
                                let mut guard = samples.lock().unwrap();
                                std::mem::take(&mut *guard)
                            };
                            let cfg_clone = cfg.clone();
                            let trans = Arc::clone(&transcriber);
                            let ui_tx = ui_tx.clone();

                            tokio::spawn(async move {
                                if pcm.is_empty() {
                                    let _ = ui_tx
                                        .send(AppMessage::Status("⚠️ No audio recorded".into()))
                                        .await;
                                    return;
                                }

                                let wav_bytes = tokio::task::spawn_blocking(move || -> Vec<u8> {
                                    let mut buf = Vec::<u8>::with_capacity(pcm.len() * 4 + 44);
                                    let spec = hound::WavSpec {
                                        channels: 1,
                                        sample_rate: cfg_clone.sample_rate().0,
                                        bits_per_sample: 32,
                                        sample_format: hound::SampleFormat::Float,
                                    };
                                    let mut writer = hound::WavWriter::new(Cursor::new(&mut buf), spec)
                                        .expect("wav writer");
                                    for s in pcm {
                                        writer.write_sample(s).unwrap();
                                    }
                                    writer.finalize().unwrap();
                                    buf
                                })
                                .await
                                .unwrap();

                                let result_msg = match trans
                                    .chat(&[ChatMessage::user().audio(wav_bytes).build()])
                                    .await
                                {
                                    Ok(_) => "✅ Audio processed".to_string(),
                                    Err(e) => format!("❌ Error: {e}"),
                                };
                                
                                let _ = ui_tx.send(AppMessage::Status(result_msg.clone())).await;
                                
                                let success = result_msg.starts_with("✅");
                                
                                if success {
                                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                    let _ = ui_tx
                                        .send(AppMessage::Status(
                                            "Press SPACE to record, 'q' to quit".into(),
                                        ))
                                        .await;
                                }
                            });
                        } else {
                            samples.lock().unwrap().clear();
                            let samples_clone = Arc::clone(&samples);
                            let chans = cfg.channels() as usize;

                            let s = input_dev.build_input_stream(
                                &cfg.clone().into(),
                                move |data: &[f32], _| {
                                    let mut buf = samples_clone.lock().unwrap();
                                    buf.extend(data.iter().step_by(chans));
                                },
                                |err| eprintln!("Audio error: {err}"),
                                None,
                            )?;
                            s.play()?;
                            stream = Some(s);
                            app.set_recording(true);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}
