use cpal::traits::HostTrait;
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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

    tokio::spawn(i2c_imu());

    Ok(())

    // info!("Starting Whisper Live Speech Recognition");

    // let host = cpal::default_host();
    // let device = host
    //     .default_input_device()
    //     .ok_or(Error::NoInputDeviceAvailable)?;

    // let (text_receiver, stream) = whisper_realtime(
    //     "models/whisper-small-encoder.onnx",
    //     "models/whisper-small-decoder.onnx",
    //     "models/whisper-small-tokenizer.json",
    //     device,
    // )?;

    // while let Ok(text) = text_receiver.recv() {
    //     info!("Transcription: {}", text);
    // }

    // #[allow(unreachable_code)]
    // {
    //     drop(stream);
    //     Ok(())
    // }
}

#[derive(thiserror::Error, Debug)]
enum Error {
    #[error("ONNX Runtime error: {0}")]
    OnnxRuntimeError(#[from] ort::Error),

    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),

    #[error("Default input device error: {0}")]
    DefaultInputDeviceError(#[from] cpal::DefaultStreamConfigError),

    #[error("No input device available")]
    NoInputDeviceAvailable,

    #[error("Audio device name error: {0}")]
    DeviceNameError(#[from] cpal::DeviceNameError),

    #[error("Build audio stream error: {0}")]
    BuildAudioStreamError(#[from] cpal::BuildStreamError),

    #[error("Play audio stream error: {0}")]
    PlayAudioStreamError(#[from] cpal::PlayStreamError),

    #[error("Shape error: {0}")]
    ShapeError(#[from] ndarray::ShapeError),

    #[error("Linux I2C error: {0}")]
    I2CError(#[from] i2cdev::linux::LinuxI2CError),
}
