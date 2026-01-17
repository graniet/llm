//! PTY spawn logic.

use std::io::{ErrorKind, Read, Write};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use tokio::sync::{broadcast, mpsc, Mutex as TokioMutex};
use tokio::task::JoinHandle;

use crate::tools::error::ToolError;

use super::types::{PtyPair, SessionId};
use super::PtySession;

/// Spawn a new PTY session with the given command.
pub async fn spawn_session(
    id: SessionId,
    shell: &str,
    command: &str,
    working_dir: &str,
) -> Result<(PtySession, broadcast::Receiver<Vec<u8>>), ToolError> {
    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize {
            rows: 24,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| ToolError::Execution(format!("Failed to open PTY: {e}")))?;

    let mut cmd_builder = CommandBuilder::new(shell);
    cmd_builder.arg("-c");
    cmd_builder.arg(command);
    cmd_builder.cwd(working_dir);

    let child = pair
        .slave
        .spawn_command(cmd_builder)
        .map_err(|e| ToolError::Execution(format!("Failed to spawn command: {e}")))?;

    let killer = child.clone_killer();
    let (writer_tx, mut writer_rx) = mpsc::channel::<Vec<u8>>(128);
    let (output_tx, _) = broadcast::channel::<Vec<u8>>(256);
    let initial_output_rx = output_tx.subscribe();

    // Reader task
    let reader_handle = spawn_reader(&pair, output_tx.clone())?;

    // Writer task
    let writer_handle = spawn_writer(&pair, &mut writer_rx)?;

    // Wait task
    let (exit_status, exit_code) = PtySession::default_exit_state();
    let wait_handle = spawn_wait_task(child, Arc::clone(&exit_status), Arc::clone(&exit_code));

    let pty_pair = PtyPair {
        _slave: if cfg!(windows) {
            Some(pair.slave)
        } else {
            None
        },
        _master: pair.master,
    };

    let session = PtySession {
        id,
        writer_tx,
        output_tx,
        exit_status,
        exit_code,
        reader_handle: StdMutex::new(Some(reader_handle)),
        writer_handle: StdMutex::new(Some(writer_handle)),
        wait_handle: StdMutex::new(Some(wait_handle)),
        killer: StdMutex::new(Some(killer)),
        _pair: StdMutex::new(pty_pair),
    };

    Ok((session, initial_output_rx))
}

fn spawn_reader(
    pair: &portable_pty::PtyPair,
    output_tx: broadcast::Sender<Vec<u8>>,
) -> Result<JoinHandle<()>, ToolError> {
    let mut reader = pair
        .master
        .try_clone_reader()
        .map_err(|e| ToolError::Execution(format!("Failed to clone reader: {e}")))?;

    let handle = tokio::task::spawn_blocking(move || {
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let _ = output_tx.send(buf[..n].to_vec());
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(ref e) if e.kind() == ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                }
                Err(_) => break,
            }
        }
    });

    Ok(handle)
}

fn spawn_writer(
    pair: &portable_pty::PtyPair,
    writer_rx: &mut mpsc::Receiver<Vec<u8>>,
) -> Result<JoinHandle<()>, ToolError> {
    let writer = pair
        .master
        .take_writer()
        .map_err(|e| ToolError::Execution(format!("Failed to take writer: {e}")))?;
    let writer = Arc::new(TokioMutex::new(writer));

    // Take ownership of receiver for the task
    let (_new_tx, mut new_rx) = mpsc::channel::<Vec<u8>>(128);
    std::mem::swap(writer_rx, &mut new_rx);

    let handle = tokio::spawn({
        async move {
            while let Some(bytes) = new_rx.recv().await {
                let mut guard = writer.lock().await;
                let _ = guard.write_all(&bytes);
                let _ = guard.flush();
            }
        }
    });

    Ok(handle)
}

fn spawn_wait_task(
    mut child: Box<dyn portable_pty::Child + Send + Sync>,
    exit_status: Arc<std::sync::atomic::AtomicBool>,
    exit_code: Arc<StdMutex<Option<i32>>>,
) -> JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        let code = match child.wait() {
            Ok(status) => status.exit_code() as i32,
            Err(_) => -1,
        };
        exit_status.store(true, Ordering::SeqCst);
        if let Ok(mut guard) = exit_code.lock() {
            *guard = Some(code);
        }
    })
}
