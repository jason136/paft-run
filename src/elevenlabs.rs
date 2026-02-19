use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::process::{Child, Command};
use tracing::{info, warn};

use crate::imu::Imu;
use crate::servo::Servos;
use crate::Error;

const SOCKET_PATH: &str = "/tmp/paft-agent.sock";

pub struct Agent {
    process: Child,
    tool_handler_handle: JoinHandle<Result<(), Error>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "method")]
enum Request {
    DetectImage,
}

#[derive(Serialize)]
struct Response {
    error: Option<String>,
}

impl Agent {
    pub async fn spawn() -> Result<Self, Error> {
        let _ = std::fs::remove_file(SOCKET_PATH);

        let listener = UnixListener::bind(SOCKET_PATH)?;
        info!("Agent bridge listening on {}", SOCKET_PATH);

        let tool_handler_handle = tokio::spawn(async move {
            loop {
                let (stream, _addr) = listener.accept().await?;
                info!("Agent connected");

                let (reader, mut writer) = stream.into_split();
                let mut lines = BufReader::new(reader).lines();

                while let Ok(Some(line)) = lines.next_line().await {
                    let Ok(request) = serde_json::from_str::<Request>(&line) else {
                        tracing::warn!("Invalid request: {line}");
                        continue;
                    };

                    match request {
                        Request::DetectImage => {
                            tracing::warn!("Detecting image not implemented");
                        }
                    }
                }

                info!("Agent disconnected");
            }
        });

        let process = Command::new("elevenlabs").arg("tool-handler").spawn()?;

        Ok(Agent {
            process,
            tool_handler_handle,
        })
    }
}
