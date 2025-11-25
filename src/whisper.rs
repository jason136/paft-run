use cpal::traits::{DeviceTrait, StreamTrait};
use flume::Receiver;
use ndarray::{s, Array2, Array3, Axis};
use ort::{
    execution_providers::QNNExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
use ringbuf::{
    traits::{Consumer, Producer, Split},
    HeapRb,
};
use rustfft::{num_complex::Complex, FftPlanner};
use tokenizers::Tokenizer;
use tracing::{info, warn};

use crate::Error;

// Whisper model constants
const SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const CHUNK_LENGTH: usize = 30; // seconds
const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE as usize; // 30 seconds of audio

// Decoder constants
const N_TEXT_CTX: usize = 448;
const SOT_TOKEN: i64 = 50258;
const EOT_TOKEN: i64 = 50257;
const TRANSCRIBE_TOKEN: i64 = 50359;
const NO_TIMESTAMPS_TOKEN: i64 = 50363;
const ENGLISH_TOKEN: i64 = 50259;

/// Whisper ONNX model wrapper
struct WhisperModel {
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
}

impl WhisperModel {
    fn new(encoder_path: &str, decoder_path: &str, tokenizer_path: &str) -> Result<Self, Error> {
        info!("Initializing ONNX Runtime...");
        ort::init()
            .with_name("whisper")
            .with_execution_providers([QNNExecutionProvider::default().build()])
            .commit()?;

        info!("Loading encoder from: {}", encoder_path);
        let encoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(encoder_path)?;

        info!("Loading decoder from: {}", decoder_path);
        let decoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(decoder_path)?;

        info!("Models loaded successfully!");

        // Print model info
        info!(
            "Encoder inputs: {:?}",
            encoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
        );
        info!(
            "Encoder outputs: {:?}",
            encoder.outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
        );
        info!(
            "Decoder inputs: {:?}",
            decoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>()
        );
        info!(
            "Decoder outputs: {:?}",
            decoder.outputs.iter().map(|o| &o.name).collect::<Vec<_>>()
        );

        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
        })
    }

    /// Run encoder on mel spectrogram features
    fn encode(&self, mel: Array2<f32>) -> Result<Array3<f32>, Error> {
        // Reshape mel to [1, n_mels, time_steps]
        let mel_3d = mel.insert_axis(Axis(0));

        info!("Running encoder with input shape: {:?}", mel_3d.shape());

        let outputs = self.encoder.run(ort::inputs![mel_3d])?;
        let audio_features = outputs[0].try_extract_tensor::<f32>()?;

        Ok(audio_features.view().to_owned().into_dimensionality()?)
    }

    /// Run decoder to generate tokens
    fn decode(
        &self,
        audio_features: &Array3<f32>,
        tokens: &[i64],
    ) -> Result<(Vec<f32>, i64), Error> {
        // Prepare input tokens - shape [1, seq_len]
        let tokens_array = Array2::from_shape_vec((1, tokens.len()), tokens.to_vec())?;

        let outputs = self.decoder.run(ort::inputs![
            "audio_features" => audio_features.view(),
            "tokens" => tokens_array.view(),
        ]?)?;

        let logits = outputs[0].try_extract_tensor::<f32>()?;
        let logits_view = logits.view();

        // Get logits for the last token
        let last_token_logits = logits_view.slice(s![0, -1, ..]).to_owned();

        // Find the token with highest probability
        let next_token = last_token_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as i64)
            .unwrap();

        Ok((last_token_logits.to_vec(), next_token))
    }

    /// Transcribe audio samples
    fn transcribe(&self, audio: &[f32]) -> Result<String, Error> {
        info!("Transcribing {} samples...", audio.len());

        // Convert audio to mel spectrogram
        let mel = audio_to_mel_spectrogram(audio)?;
        info!("Mel spectrogram shape: {:?}", mel.shape());

        // Encode audio
        let audio_features = self.encode(mel)?;
        info!("Audio features shape: {:?}", audio_features.shape());

        // Initialize decoder with start tokens
        let mut tokens = vec![
            SOT_TOKEN,
            ENGLISH_TOKEN,
            TRANSCRIBE_TOKEN,
            NO_TIMESTAMPS_TOKEN,
        ];

        info!("Starting decoding...");

        // Decode tokens autoregressively
        let max_tokens = 50;
        for i in 0..max_tokens {
            let (_, next_token) = self.decode(&audio_features, &tokens)?;

            if next_token == EOT_TOKEN {
                info!("Found EOT token at position {}", i);
                break;
            }

            tokens.push(next_token);
        }

        let text = self.tokenizer.decode(
            &tokens[4..].iter().map(|t| *t as u32).collect::<Vec<_>>(),
            true,
        )?;

        Ok(text)
    }
}

/// Convert audio samples to mel spectrogram
fn audio_to_mel_spectrogram(audio: &[f32]) -> Result<Array2<f32>, Error> {
    let n_frames = (audio.len() - N_FFT) / HOP_LENGTH + 1;

    // Pad or truncate audio to expected length
    let mut padded_audio = audio.to_vec();
    if padded_audio.len() < N_SAMPLES {
        padded_audio.resize(N_SAMPLES, 0.0);
    } else if padded_audio.len() > N_SAMPLES {
        padded_audio.truncate(N_SAMPLES);
    }

    let expected_frames = (N_SAMPLES - N_FFT) / HOP_LENGTH + 1;

    // Create STFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let mut spectrogram = Array2::<f32>::zeros((N_FFT / 2 + 1, expected_frames));

    for i in 0..expected_frames {
        let start = i * HOP_LENGTH;
        let end = start + N_FFT;

        if end > padded_audio.len() {
            break;
        }

        // Apply Hann window
        let mut windowed: Vec<Complex<f32>> = padded_audio[start..end]
            .iter()
            .enumerate()
            .map(|(j, &x)| {
                let window = 0.5
                    * (1.0 - (2.0 * std::f32::consts::PI * j as f32 / (N_FFT - 1) as f32).cos());
                Complex::new(x * window, 0.0)
            })
            .collect();

        // Compute FFT
        fft.process(&mut windowed);

        // Take magnitude
        for (j, val) in windowed.iter().take(N_FFT / 2 + 1).enumerate() {
            spectrogram[[j, i]] = val.norm();
        }
    }

    // Convert to mel scale (simplified)
    let mel = spectrogram_to_mel(&spectrogram)?;

    // Apply log
    let mel_log = mel.mapv(|x| (x.max(1e-10)).ln());

    // Normalize
    let max_val = mel_log.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mel_normalized = mel_log.mapv(|x| (x.max(max_val - 8.0) + 4.0) / 4.0);

    Ok(mel_normalized)
}

/// Convert spectrogram to mel scale (simplified linear approximation)
fn spectrogram_to_mel(spec: &Array2<f32>) -> Result<Array2<f32>, Error> {
    let n_fft = (spec.shape()[0] - 1) * 2;
    let n_frames = spec.shape()[1];

    // Create mel filterbank (simplified)
    let mut mel = Array2::<f32>::zeros((N_MELS, n_frames));

    let mel_bins_per_freq_bin = spec.shape()[0] as f32 / N_MELS as f32;

    for i in 0..N_MELS {
        for j in 0..n_frames {
            // Simple averaging across frequency bins
            let start_bin = (i as f32 * mel_bins_per_freq_bin) as usize;
            let end_bin = ((i + 1) as f32 * mel_bins_per_freq_bin) as usize;
            let end_bin = end_bin.min(spec.shape()[0]);

            let mut sum = 0.0;
            let mut count = 0;
            for k in start_bin..end_bin {
                sum += spec[[k, j]];
                count += 1;
            }
            mel[[i, j]] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }

    Ok(mel)
}

pub fn whisper_realtime(
    encoder_path: &str,
    decoder_path: &str,
    tokenizer_path: &str,
    audio_device: cpal::Device,
) -> Result<Receiver<String>, Error> {
    let config = audio_device.default_input_config()?;
    info!("Default input config: {:?}", config);

    let desired_config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    // let (audio_sender, audio_receiver) = flume::unbounded();

    let buffer = HeapRb::new(N_SAMPLES);
    let (mut audio_producer, mut audio_consumer) = buffer.split();

    let stream = audio_device.build_input_stream(
        &desired_config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            audio_producer.push_slice(data);
        },
        |err| {
            warn!("Audio stream error: {}", err);
        },
        None,
    )?;

    let model = WhisperModel::new(encoder_path, decoder_path, tokenizer_path)?;

    stream.play()?;

    let (audio_chunk_sender, audio_chunk_receiver) = flume::unbounded();
    tokio::task::spawn(async move {
        let mut chunk = vec![0.0; N_SAMPLES];

        loop {
            chunk.clear();
            audio_consumer.peek_slice(&mut chunk);

            if chunk.iter().map(|x| x * x).sum::<f32>() > 100.0 {
                audio_chunk_sender.send(chunk.clone());
                audio_consumer.clear();
            }

            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    });

    let (text_sender, text_receiver) = flume::unbounded();

    tokio::task::spawn_blocking(move || {
        while let Ok(chunk) = audio_chunk_receiver.recv() {
            match model.transcribe(&chunk) {
                Ok(text) => {
                    text_sender.send(text);
                    continue;
                }
                Err(e) => warn!("Error transcribing chunk: {}", e),
            };
        }
    });

    info!("Audio stream started");

    Ok(text_receiver)
}
