use std::time::{Duration, Instant};

use crossterm::event::Event;
use tokio::sync::mpsc;

use crate::runtime::{AppEvent, InputEvent};
use crate::ui::{render_app, MessageRenderer};

use super::controller::AppController;
use super::terminal::AppTerminal;

pub async fn run_app(
    controller: AppController,
    terminal: &mut AppTerminal,
    rx: mpsc::Receiver<AppEvent>,
    event_tx: mpsc::Sender<AppEvent>,
) -> anyhow::Result<()> {
    let mut runner = AppRunner::new(controller, terminal, rx, event_tx);
    runner.run().await
}

struct AppRunner<'a> {
    controller: AppController,
    terminal: &'a mut AppTerminal,
    rx: mpsc::Receiver<AppEvent>,
    renderer: MessageRenderer,
    tick: tokio::time::Interval,
    dirty: bool,
}

const SLOW_FRAME_MS: u64 = 100;

impl<'a> AppRunner<'a> {
    fn new(
        mut controller: AppController,
        terminal: &'a mut AppTerminal,
        rx: mpsc::Receiver<AppEvent>,
        event_tx: mpsc::Sender<AppEvent>,
    ) -> Self {
        controller.state.terminal_size = terminal
            .size()
            .map(|r| (r.width, r.height))
            .unwrap_or((0, 0));
        spawn_event_reader(event_tx);
        Self {
            controller,
            terminal,
            rx,
            renderer: MessageRenderer::default(),
            tick: tokio::time::interval(Duration::from_millis(50)),
            dirty: true,
        }
    }

    async fn run(&mut self) -> anyhow::Result<()> {
        while !self.controller.state.should_quit {
            let dirty = self.wait_for_event().await?;
            if dirty {
                self.draw()?;
                self.dirty = false;
            }
        }
        Ok(())
    }

    async fn wait_for_event(&mut self) -> anyhow::Result<bool> {
        tokio::select! {
            Some(event) = self.rx.recv() => {
                if self.controller.handle_event(event).await {
                    self.dirty = true;
                }
            }
            _ = self.tick.tick() => {
                if self.controller.handle_event(AppEvent::Tick).await {
                    self.dirty = true;
                }
            }
        }
        Ok(self.dirty)
    }

    fn draw(&mut self) -> anyhow::Result<()> {
        let started = Instant::now();
        self.terminal
            .draw(|frame| render_app(frame, &self.controller.state, &mut self.renderer))?;
        if started.elapsed() > Duration::from_millis(SLOW_FRAME_MS) {
            self.controller.state.animation.disable();
        }
        Ok(())
    }
}

fn spawn_event_reader(sender: mpsc::Sender<AppEvent>) {
    std::thread::spawn(move || loop {
        let event = match crossterm::event::read() {
            Ok(ev) => ev,
            Err(_) => continue,
        };
        let mapped = match event {
            Event::Key(key) => Some(InputEvent::Key(key)),
            Event::Mouse(mouse) => Some(InputEvent::Mouse(mouse)),
            Event::Paste(text) => Some(InputEvent::Paste(text)),
            Event::Resize(w, h) => Some(InputEvent::Resize(w, h)),
            _ => None,
        };
        if let Some(input) = mapped {
            let _ = sender.blocking_send(AppEvent::Input(input));
        }
    });
}
