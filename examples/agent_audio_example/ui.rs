use anyhow::{Context, Result};
use crossterm::{
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
use std::collections::VecDeque;
use std::io;

const STATUS_HEIGHT: u16 = 3;
const MAX_MESSAGES: usize = 512;
const TITLE_MESSAGES: &str = "Agent messages";
const TITLE_CONTROLS: &str = "Controls";

pub struct AppState {
    messages: VecDeque<String>,
    status: String,
    recording: bool,
}

impl AppState {
    pub fn new(initial_status: impl Into<String>) -> Self {
        Self {
            messages: VecDeque::new(),
            status: initial_status.into(),
            recording: false,
        }
    }

    pub fn push_message(&mut self, role: &str, content: &str) {
        if self.messages.len() == MAX_MESSAGES {
            self.messages.pop_front();
        }
        let msg = format!("[{role}] {content}");
        self.messages.push_back(msg);
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    pub fn is_recording(&self) -> bool {
        self.recording
    }

    pub fn set_recording(&mut self, recording: bool) {
        self.recording = recording;
    }
}

pub struct TerminalSession {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalSession {
    pub fn new() -> Result<Self> {
        enable_raw_mode().context("failed to enable raw mode")?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend).context("failed to create terminal")?;
        Ok(Self { terminal })
    }

    pub fn draw(&mut self, app: &AppState) -> Result<()> {
        self.terminal.draw(|frame| render(frame, app))?;
        Ok(())
    }

    pub fn restore(&mut self) -> Result<()> {
        disable_raw_mode().context("failed to disable raw mode")?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)
            .context("failed to leave alternate screen")?;
        self.terminal
            .show_cursor()
            .context("failed to show cursor")?;
        Ok(())
    }
}

fn render(frame: &mut Frame, app: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(STATUS_HEIGHT)])
        .split(frame.area());

    let items: Vec<ListItem> = app
        .messages
        .iter()
        .map(|m| ListItem::new(m.as_str()))
        .collect();

    frame.render_widget(
        List::new(items).block(Block::default().title(TITLE_MESSAGES).borders(Borders::ALL)),
        chunks[0],
    );

    let style = if app.recording {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Green)
    };

    frame.render_widget(
        Paragraph::new(app.status.as_str())
            .block(Block::default().title(TITLE_CONTROLS).borders(Borders::ALL))
            .style(style),
        chunks[1],
    );
}
