use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tracing::info;

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
    UseVision,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "result")]
enum Response {
    Success { value: serde_json::Value },
    Error { message: String },
}

impl Agent {
    pub async fn spawn() -> Result<Self, Error> {
        let _ = std::fs::remove_file(SOCKET_PATH);

        let listener = UnixListener::bind(SOCKET_PATH)?;
        info!("Agent bridge listening on {}", SOCKET_PATH);

        let tool_handler_handle = tokio::spawn(async move {
            loop {
                let (mut stream, _addr) = listener.accept().await?;
                info!("Agent tool connection");

                tokio::spawn(async move {
                    let (reader, mut writer) = stream.split();
                    let mut lines = BufReader::new(reader).lines();

                    while let Ok(Some(line)) = lines.next_line().await {
                        let response = match serde_json::from_str::<Request>(&line) {
                            Ok(request) => match request {
                                Request::UseVision => Response::Success {
                                    value: serde_json::json!({"detected": []}),
                                },
                            },
                            Err(e) => Response::Error {
                                message: format!("invalid request: {e}"),
                            },
                        };

                        let mut buf = serde_json::to_vec(&response).unwrap();
                        buf.push(b'\n');
                        let _ = writer.write_all(&buf).await;
                    }
                });
            }
        });

        let process = Command::new("uv")
            .arg("run")
            .arg("src/python/elevenlabs.py")
            .spawn()?;

        Ok(Agent {
            process,
            tool_handler_handle,
        })
    }
}
