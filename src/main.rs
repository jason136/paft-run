use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ndarray::{s, Array2, Array3, Axis};
use ringbuf::HeapRb;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod imu;
mod whisper;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with("info,ort=debug".into())
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting Whisper Live Speech Recognition");

    // Load models - adjust paths to your actual model locations
    let model = WhisperModel::new(
        "models/whisper-small-encoder.onnx",
        "models/whisper-small-decoder.onnx",
    )?;

    // Start audio capture
    let audio_capture = AudioCapture::new(SAMPLE_RATE as usize * 60); // 60 seconds buffer
    let stream = audio_capture.start()?;

    info!("Listening... Speak into your microphone!");
    info!("Processing audio in 30-second chunks");

    // Wait for initial buffer to fill
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    loop {
        // Wait until we have enough samples
        while audio_capture.available_samples() < N_SAMPLES {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        // Read audio chunk
        let audio_chunk = audio_capture.read_samples(N_SAMPLES);

        // Check if there's actual speech (simple energy-based VAD)
        let energy: f32 =
            audio_chunk.iter().map(|&x| x * x).sum::<f32>() / audio_chunk.len() as f32;
        let threshold = 0.001;

        if energy > threshold {
            info!("Speech detected (energy: {:.6}), transcribing...", energy);

            // Transcribe in a blocking task to not block the audio thread
            let audio_for_transcription = audio_chunk.clone();
            let transcription =
                tokio::task::spawn_blocking(move || model.transcribe(&audio_for_transcription))
                    .await??;

            info!("Transcription: {}", transcription);
            println!("\n>>> {}\n", transcription);
        } else {
            info!("Silence detected (energy: {:.6}), skipping...", energy);
        }

        // Small delay to prevent tight loop
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    #[allow(unreachable_code)]
    {
        drop(stream);
        Ok(())
    }
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
}
