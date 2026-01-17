use crate::app::AppContext;
use crate::audio::{pcm_to_wav_bytes, AudioRecorder};
use crate::ui::{AppState, TerminalSession};
use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode};
use llm::{
    chat::ChatMessage,
    memory::{SharedMemory, SlidingWindowMemory},
    LLMProvider,
};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Handle;
const STATUS_IDLE: &str = "Press SPACE to record, 'q' to quit";
const STATUS_RECORDING: &str = "Recording... press SPACE to stop";
const STATUS_PROCESSING: &str = "Processing audio...";
const STATUS_NO_AUDIO: &str = "No audio recorded";
const STATUS_OK: &str = "Audio processed";
const STATUS_ERROR_PREFIX: &str = "Error: ";
const STATUS_RESET_DELAY: Duration = Duration::from_secs(2);
const UI_POLL_MS: u64 = 50;
pub fn run(ctx: AppContext, handle: Handle) -> Result<()> {
    let (ui_tx, ui_rx) = mpsc::channel::<AppMessage>();
    spawn_memory_listener(&handle, ctx.memory.clone(), ui_tx.clone())?;
    let services = ProcessingContext::new(handle, Arc::clone(&ctx.transcriber), ui_tx);
    let terminal = TerminalSession::new()?;
    let recorder = AudioRecorder::new()?;
    let app = AppState::new(STATUS_IDLE);
    let mut ui_ctx = UiContext {
        app,
        terminal,
        recorder,
        rx: ui_rx,
        services,
    };
    let result = ui_ctx.run_loop();
    let restore = ui_ctx.restore_terminal();
    restore?;
    result
}
fn spawn_memory_listener(
    handle: &Handle,
    memory: SharedMemory<SlidingWindowMemory>,
    ui_tx: Sender<AppMessage>,
) -> Result<()> {
    let mut receiver = memory.subscribe().context("reactive memory not enabled")?;
    handle.spawn(async move {
        while let Ok(event) = receiver.recv().await {
            if ui_tx
                .send(AppMessage::Agent {
                    role: event.role,
                    content: event.msg.content,
                })
                .is_err()
            {
                break;
            }
        }
    });
    Ok(())
}
enum AppMessage {
    Agent { role: String, content: String },
    Status(String),
}
struct ProcessingContext {
    handle: Handle,
    transcriber: Arc<dyn LLMProvider>,
    ui_tx: Sender<AppMessage>,
}
impl ProcessingContext {
    fn new(handle: Handle, transcriber: Arc<dyn LLMProvider>, ui_tx: Sender<AppMessage>) -> Self {
        Self {
            handle,
            transcriber,
            ui_tx,
        }
    }

    fn spawn_transcription(&self, samples: Vec<f32>, sample_rate: u32) {
        let transcriber = Arc::clone(&self.transcriber);
        let ui_tx = self.ui_tx.clone();
        self.handle
            .spawn(async move { process_audio(samples, sample_rate, transcriber, ui_tx).await });
    }
}
struct UiContext {
    app: AppState,
    terminal: TerminalSession,
    recorder: AudioRecorder,
    rx: Receiver<AppMessage>,
    services: ProcessingContext,
}
impl UiContext {
    fn run_loop(&mut self) -> Result<()> {
        loop {
            self.drain_messages();
            self.terminal.draw(&self.app)?;
            if self.handle_input()? {
                break;
            }
        }
        Ok(())
    }
    fn restore_terminal(&mut self) -> Result<()> {
        self.terminal.restore()
    }
    fn drain_messages(&mut self) {
        for msg in self.rx.try_iter() {
            match msg {
                AppMessage::Agent { role, content } => self.app.push_message(&role, &content),
                AppMessage::Status(status) => self.app.set_status(status),
            }
        }
    }
    fn handle_input(&mut self) -> Result<bool> {
        if !event::poll(Duration::from_millis(UI_POLL_MS))? {
            return Ok(false);
        }
        let Event::Key(key) = event::read()? else {
            return Ok(false);
        };
        match key.code {
            KeyCode::Char('q') => Ok(true),
            KeyCode::Char(' ') => {
                self.toggle_recording()?;
                Ok(false)
            }
            _ => Ok(false),
        }
    }
    fn toggle_recording(&mut self) -> Result<()> {
        if self.app.is_recording() {
            return self.stop_recording();
        }
        self.start_recording()
    }
    fn start_recording(&mut self) -> Result<()> {
        self.recorder.clear();
        self.recorder.start()?;
        self.app.set_recording(true);
        self.app.set_status(STATUS_RECORDING);
        Ok(())
    }
    fn stop_recording(&mut self) -> Result<()> {
        let samples = self.recorder.stop();
        self.app.set_recording(false);
        if samples.is_empty() {
            self.app.set_status(STATUS_NO_AUDIO);
            return Ok(());
        }
        self.app.set_status(STATUS_PROCESSING);
        self.services
            .spawn_transcription(samples, self.recorder.sample_rate());
        Ok(())
    }
}
fn send_status(tx: &Sender<AppMessage>, status: String) -> bool {
    tx.send(AppMessage::Status(status)).is_ok()
}
async fn process_audio(
    samples: Vec<f32>,
    sample_rate: u32,
    transcriber: Arc<dyn LLMProvider>,
    ui_tx: Sender<AppMessage>,
) {
    if samples.is_empty() {
        send_status(&ui_tx, STATUS_NO_AUDIO.to_string());
        return;
    }
    let wav = match pcm_to_wav_bytes(samples, sample_rate).await {
        Ok(bytes) => bytes,
        Err(err) => {
            send_status(&ui_tx, format!("{STATUS_ERROR_PREFIX}{err}"));
            return;
        }
    };
    if let Err(err) = transcribe_once(transcriber, wav).await {
        send_status(&ui_tx, format!("{STATUS_ERROR_PREFIX}{err}"));
        return;
    }
    if !send_status(&ui_tx, STATUS_OK.to_string()) {
        return;
    }
    tokio::time::sleep(STATUS_RESET_DELAY).await;
    send_status(&ui_tx, STATUS_IDLE.to_string());
}
async fn transcribe_once(
    transcriber: Arc<dyn LLMProvider>,
    wav: Vec<u8>,
) -> std::result::Result<(), llm::error::LLMError> {
    let msg = ChatMessage::user().audio(wav).build();
    transcriber.chat(&[msg]).await.map(|_| ())
}
