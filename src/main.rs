use cpal::traits::{DeviceTrait, HostTrait};
use tokio::time::sleep;
use tracing::info;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

use crate::{imu::Imu, whisper::whisper_realtime};

mod imu;
mod servo;
mod whisper;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();

    tracing::subscriber::set_global_default(subscriber).unwrap();

    tokio::spawn(async move {
        let mut imu = Imu::new().await?;

        while let Ok((acc, gyr)) = imu.sample() {
            println!("Acc: {:?}, Gyr: {:?}", acc, gyr);

            sleep(std::time::Duration::from_millis(100)).await;
        }

        Ok::<_, Error>(())
    });

    info!("Starting Whisper Live Speech Recognition");

    let host = cpal::default_host();
    let device = host
        .devices()
        .unwrap()
        .into_iter()
        .find(|d| d.name().unwrap().contains("pipewire"))
        .ok_or(Error::NoInputDeviceAvailable)?;

    let devices = host.devices();

    for dev in devices.unwrap() {
        println!("Device: {}", dev.name()?);
    }

    // println!("Using input device: {}", device.name()?);
    // println!("Using input device: {:?}", device.default_input_config()?);
    // println!("Using input device: {:?}", device.default_output_config()?);

    // return Ok(());

    let (text_receiver, stream) = whisper_realtime(
        "models/whisper-small-quantized/encoder-onnx/model.onnx",
        "models/whisper-small-quantized/decoder-onnx/model.onnx",
        "models/whisper-small-quantized/tokenizer.json",
        device,
    )?;

    let transcription = tokio::spawn(async move {
        while let Ok(text) = text_receiver.recv_async().await {
            tracing::warn!("Transcription: {}", text);
        }
    });

    transcription.await?;

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

    #[error("GPIO error: {0}")]
    GPIO(#[from] gpio_cdev::Error),
}
