use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use cpal::traits::HostTrait;

use crate::imu::i2c_imu;
use crate::whisper::whisper_realtime;

mod imu;
mod whisper;

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::registry()
        .with(LevelFilter::INFO)
        .with(tracing_subscriber::fmt::layer())
        .init();

    let imu = tokio::spawn(i2c_imu());

    info!("Starting Whisper Live Speech Recognition");

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(Error::NoInputDeviceAvailable)?;

    let (text_receiver, stream) = whisper_realtime(
        "models/whisper-small-quantized/encoder-onnx/model.onnx",
        "models/whisper-small-quantized/decoder-onnx/model.onnx",
        "models/whisper-small-quantized/tokenizer.json",
        device,
    )?;

    let transcription = tokio::spawn(async move {
        while let Ok(text) = text_receiver.recv_async().await {
            info!("Transcription: {}", text);
        }
    });

    let _ = tokio::join!(imu, transcription);

    {
        drop(stream);
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
enum Error {
    #[error("Task join error: {0}")]
    Join(#[from] tokio::task::JoinError),

    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(#[from] ort::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),

    #[error("Default input device error: {0}")]
    DefaultInputDevice(#[from] cpal::DefaultStreamConfigError),

    #[error("No input device available")]
    NoInputDeviceAvailable,

    #[error("Audio device name error: {0}")]
    DeviceName(#[from] cpal::DeviceNameError),

    #[error("Build audio stream error: {0}")]
    BuildAudioStream(#[from] cpal::BuildStreamError),

    #[error("Play audio stream error: {0}")]
    PlayAudioStream(#[from] cpal::PlayStreamError),

    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("Linux I2C error: {0}")]
    I2C(#[from] i2cdev::linux::LinuxI2CError),
}

