use std::sync::{Arc, Mutex};

use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers};
use futures::StreamExt;

use crate::actuators::Actuators;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DriveMode {
    Forward,
    Stop,
    Backward,
}

impl DriveMode {
    fn speed_pair(self) -> (i32, i32) {
        match self {
            DriveMode::Forward => (90, 90),
            DriveMode::Stop => (0, 0),
            DriveMode::Backward => (-90, -90),
        }
    }
}

pub fn spawn_stdin_teleop(actuators: Actuators) -> tokio::task::JoinHandle<()> {
    let actuators = Arc::new(Mutex::new(actuators));

    tokio::spawn(async move {
        struct RawModeGuard;

        impl Drop for RawModeGuard {
            fn drop(&mut self) {
                let _ = crossterm::terminal::disable_raw_mode();
            }
        }

        crossterm::terminal::enable_raw_mode().unwrap();
        let _guard = RawModeGuard;

        eprintln!("Teleop: W = forward, E = stop, R = backward (both motors). Q/Esc = quit");

        let mut stream = EventStream::new();
        let mut mode: Option<DriveMode> = None;

        while let Some(ev) = stream.next().await {
            let Ok(ev) = ev else { break };

            let Event::Key(key) = ev else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }

            let next = match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                KeyCode::Char('w') => Some(DriveMode::Forward),
                KeyCode::Char('e') => Some(DriveMode::Stop),
                KeyCode::Char('r') => Some(DriveMode::Backward),
                _ => None,
            };

            let Some(next) = next else { continue };
            if mode == Some(next) {
                continue;
            }

            let (l, r) = next.speed_pair();
            let ac = Arc::clone(&actuators);
            match tokio::task::spawn_blocking(move || ac.lock().expect("actuators").actuate(l, r))
                .await
            {
                Ok(Ok(())) => mode = Some(next),
                Ok(Err(e)) => tracing::warn!("actuate: {e}"),
                Err(e) => tracing::warn!("spawn_blocking: {e}"),
            }
        }
    })
}
