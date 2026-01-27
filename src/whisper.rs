use std::sync::{Arc, Mutex};

use cpal::{
    traits::{DeviceTrait, StreamTrait},
    Stream,
};
use flume::Receiver;
use half::f16;
use ndarray::{Array2, Array4, Axis};
use ort::{
    execution_providers::QNNExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use ringbuf::{
    traits::{Consumer, RingBuffer},
    HeapRb,
};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
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
const N_FRAMES: usize = 3000; // Qualcomm model expects exactly 80x3000

// Decoder constants
const SOT_TOKEN: i64 = 50258;
const EOT_TOKEN: i64 = 50257;
const TRANSCRIBE_TOKEN: i64 = 50359;
const NO_TIMESTAMPS_TOKEN: i64 = 50363;
const ENGLISH_TOKEN: i64 = 50259;

// Decoder architecture constants
const NUM_LAYERS: usize = 12;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const MAX_SEQ_LEN: usize = 200;
const ENCODER_SEQ_LEN: usize = 1500;

/// Holds KV cache state for autoregressive decoding
struct DecoderState {
    /// Self-attention key caches for each layer [12, 1, 64, 199]
    k_cache_self: Vec<Array4<u8>>,
    /// Self-attention value caches for each layer [12, 1, 199, 64]
    v_cache_self: Vec<Array4<u8>>,
    /// Cross-attention key caches (from encoder) [12, 1, 64, 1500]
    k_cache_cross: Vec<Array4<u8>>,
    /// Cross-attention value caches (from encoder) [12, 1, 1500, 64]
    v_cache_cross: Vec<Array4<u8>>,
    /// Current position in sequence
    position: i32,
}

impl DecoderState {
    fn new(k_caches: Vec<Array4<u8>>, v_caches: Vec<Array4<u8>>) -> Self {
        // Initialize self-attention caches with zeros
        let k_cache_self: Vec<_> = (0..NUM_LAYERS)
            .map(|_| Array4::<u8>::zeros((NUM_HEADS, 1, HEAD_DIM, MAX_SEQ_LEN - 1)))
            .collect();
        let v_cache_self: Vec<_> = (0..NUM_LAYERS)
            .map(|_| Array4::<u8>::zeros((NUM_HEADS, 1, MAX_SEQ_LEN - 1, HEAD_DIM)))
            .collect();

        Self {
            k_cache_self,
            v_cache_self,
            k_cache_cross: k_caches,
            v_cache_cross: v_caches,
            position: 0,
        }
    }
}

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
    /// Returns (k_cache_cross, v_cache_cross) for each layer
    fn encode(&mut self, mel: Array2<f32>) -> Result<(Vec<Array4<u8>>, Vec<Array4<u8>>), Error> {
        // Convert to f16, then to raw u16 bits (Qualcomm model expects uint16 representation of float16)
        let mel_u16 = mel.mapv(|x| f16::from_f32(x).to_bits());

        // Reshape mel to [1, n_mels, time_steps]
        let mel_3d = mel_u16.insert_axis(Axis(0));

        info!("Running encoder with input shape: {:?}", mel_3d.shape());

        let outputs = self.encoder.run(ort::inputs![Value::from_array(mel_3d)?])?;

        info!("Encoder produced {} outputs", outputs.len());

        // Encoder outputs 24 tensors: k_cache_cross_0, v_cache_cross_0, k_cache_cross_1, v_cache_cross_1, ...
        let mut k_caches = Vec::with_capacity(NUM_LAYERS);
        let mut v_caches = Vec::with_capacity(NUM_LAYERS);

        for i in 0..NUM_LAYERS {
            let k_idx = i * 2;
            let v_idx = i * 2 + 1;

            let (k_shape, k_data) = outputs[k_idx].try_extract_tensor::<u8>()?;
            let k_cache = Array4::from_shape_vec(
                (
                    k_shape[0] as usize,
                    k_shape[1] as usize,
                    k_shape[2] as usize,
                    k_shape[3] as usize,
                ),
                k_data.to_vec(),
            )?;

            let (v_shape, v_data) = outputs[v_idx].try_extract_tensor::<u8>()?;
            let v_cache = Array4::from_shape_vec(
                (
                    v_shape[0] as usize,
                    v_shape[1] as usize,
                    v_shape[2] as usize,
                    v_shape[3] as usize,
                ),
                v_data.to_vec(),
            )?;

            if i == 0 {
                info!(
                    "Layer {} k_cache shape: {:?}, v_cache shape: {:?}",
                    i,
                    k_cache.shape(),
                    v_cache.shape()
                );
            }

            k_caches.push(k_cache);
            v_caches.push(v_cache);
        }

        Ok((k_caches, v_caches))
    }

    /// Run decoder for a single token with KV cache
    fn decode_step(&mut self, token: i32, state: &mut DecoderState) -> Result<i64, Error> {
        // Prepare inputs
        let input_ids = Array2::from_shape_vec((1, 1), vec![token])?;

        // Attention mask: [1, 1, 1, 200] - mask future positions
        let mut attention_mask = Array4::<u16>::zeros((1, 1, 1, MAX_SEQ_LEN));
        // Set attended positions to 0, masked to max value (for float16 this becomes -inf after conversion)
        for i in 0..=state.position as usize {
            if i < MAX_SEQ_LEN {
                attention_mask[[0, 0, 0, i]] = 0;
            }
        }
        for i in (state.position as usize + 1)..MAX_SEQ_LEN {
            attention_mask[[0, 0, 0, i]] = 0xFC00; // -inf in float16 bits
        }

        let position_ids = ndarray::Array1::from_vec(vec![state.position]);

        // Build inputs in EXACT order from model:
        // ["input_ids", "attention_mask", self_caches..., cross_caches..., "position_ids"]
        let mut inputs: Vec<(std::borrow::Cow<str>, Value)> = vec![
            ("input_ids".into(), Value::from_array(input_ids)?.into()),
            (
                "attention_mask".into(),
                Value::from_array(attention_mask)?.into(),
            ),
        ];

        // Add all self-attention caches first
        for i in 0..NUM_LAYERS {
            inputs.push((
                format!("k_cache_self_{}_in", i).into(),
                Value::from_array(state.k_cache_self[i].clone())?.into(),
            ));
            inputs.push((
                format!("v_cache_self_{}_in", i).into(),
                Value::from_array(state.v_cache_self[i].clone())?.into(),
            ));
        }

        // Add all cross-attention caches
        for i in 0..NUM_LAYERS {
            inputs.push((
                format!("k_cache_cross_{}", i).into(),
                Value::from_array(state.k_cache_cross[i].clone())?.into(),
            ));
            inputs.push((
                format!("v_cache_cross_{}", i).into(),
                Value::from_array(state.v_cache_cross[i].clone())?.into(),
            ));
        }

        // position_ids goes LAST
        inputs.push((
            "position_ids".into(),
            Value::from_array(position_ids)?.into(),
        ));

        let outputs = self.decoder.run(inputs)?;

        // // Log output names on first call
        // if state.position == 0 {
        //     info!(
        //         "Decoder outputs: {:?}",
        //         self.decoder
        //             .outputs
        //             .iter()
        //             .map(|o| &o.name)
        //             .collect::<Vec<_>>()
        //     );
        // }

        // First output should be logits
        let (shape, data) = outputs[0].try_extract_tensor::<u16>()?;

        // Convert from raw u16 bits back to f32 for argmax
        // Logits shape is [1, 51865, 1, 1] - vocab size is 51865
        let logits: Vec<f32> = data.iter().map(|x| f16::from_bits(*x).to_f32()).collect();

        // Find the token with highest probability
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as i64)
            .unwrap_or(EOT_TOKEN);

        // // Log output names on first call to verify cache output order
        // if state.position == 0 {
        //     info!(
        //         "Decoder has {} outputs: {:?}",
        //         outputs.len(),
        //         self.decoder
        //             .outputs
        //             .iter()
        //             .map(|o| &o.name)
        //             .collect::<Vec<_>>()
        //     );
        // }

        // Update self-attention caches from decoder outputs
        // Outputs should contain k_cache_self_*_out and v_cache_self_*_out
        for i in 0..NUM_LAYERS {
            // k_cache_self output index: 1 + i*2 (after logits)
            // v_cache_self output index: 1 + i*2 + 1
            let k_idx = 1 + i * 2;
            let v_idx = 1 + i * 2 + 1;

            if k_idx < outputs.len() && v_idx < outputs.len() {
                match outputs[k_idx].try_extract_tensor::<u8>() {
                    Ok((k_shape, k_data)) => {
                        if state.position == 0 && i == 0 {
                            info!("k_cache output {} shape: {:?}", k_idx, k_shape);
                        }
                        state.k_cache_self[i] = Array4::from_shape_vec(
                            (
                                k_shape[0] as usize,
                                k_shape[1] as usize,
                                k_shape[2] as usize,
                                k_shape[3] as usize,
                            ),
                            k_data.to_vec(),
                        )
                        .unwrap_or(state.k_cache_self[i].clone());
                    }
                    Err(e) => {
                        if state.position == 0 && i == 0 {
                            warn!("Failed to extract k_cache output {}: {}", k_idx, e);
                        }
                    }
                }
                match outputs[v_idx].try_extract_tensor::<u8>() {
                    Ok((v_shape, v_data)) => {
                        if state.position == 0 && i == 0 {
                            info!("v_cache output {} shape: {:?}", v_idx, v_shape);
                        }
                        state.v_cache_self[i] = Array4::from_shape_vec(
                            (
                                v_shape[0] as usize,
                                v_shape[1] as usize,
                                v_shape[2] as usize,
                                v_shape[3] as usize,
                            ),
                            v_data.to_vec(),
                        )
                        .unwrap_or(state.v_cache_self[i].clone());
                    }
                    Err(e) => {
                        if state.position == 0 && i == 0 {
                            warn!("Failed to extract v_cache output {}: {}", v_idx, e);
                        }
                    }
                }
            }
        }

        // Update position for next iteration
        state.position += 1;

        Ok(next_token)
    }

    /// Transcribe audio samples
    fn transcribe(&mut self, audio: &[f32]) -> Result<String, Error> {
        info!("Transcribing {} samples...", audio.len());

        // Convert audio to mel spectrogram
        let mel = audio_to_mel_spectrogram(audio)?;
        info!("Mel spectrogram shape: {:?}", mel.shape());

        // Encode audio - returns per-layer k and v caches
        let (k_caches, v_caches) = self.encode(mel)?;
        info!(
            "Encoder produced {} k_caches and {} v_caches",
            k_caches.len(),
            v_caches.len()
        );

        // Initialize decoder state with encoder outputs
        let mut state = DecoderState::new(k_caches, v_caches);

        // Start tokens to feed one by one
        let start_tokens = vec![
            SOT_TOKEN as i32,
            ENGLISH_TOKEN as i32,
            TRANSCRIBE_TOKEN as i32,
            NO_TIMESTAMPS_TOKEN as i32,
        ];

        info!("Starting decoding...");

        // Feed start tokens to build up KV cache, keep the last prediction
        let mut generated_tokens: Vec<i64> = Vec::new();
        let mut next_token: i64 = SOT_TOKEN;

        for &token in &start_tokens {
            next_token = self.decode_step(token, &mut state)?;
            info!("Fed start token {}, predicted {}", token, next_token);
        }

        // Decode tokens autoregressively, starting with the prediction from last start token
        let max_tokens = 50;

        for i in 0..max_tokens {
            // Check for EOT before adding
            if next_token == EOT_TOKEN {
                info!("Found EOT token at position {}", i);
                break;
            }

            generated_tokens.push(next_token);
            info!("Generated token {}: {}", i, next_token);

            // Feed the predicted token to get the next prediction
            next_token = self.decode_step(next_token as i32, &mut state)?;
        }

        let text = self.tokenizer.decode(
            &generated_tokens
                .iter()
                .map(|t| *t as u32)
                .collect::<Vec<_>>(),
            false,
        )?;

        Ok(text)
    }
}

/// Convert audio samples to mel spectrogram
fn audio_to_mel_spectrogram(audio: &[f32]) -> Result<Array2<f32>, Error> {
    // Pad or truncate audio to expected length
    let mut padded_audio = audio.to_vec();
    if padded_audio.len() < N_SAMPLES {
        padded_audio.resize(N_SAMPLES, 0.0);
    } else if padded_audio.len() > N_SAMPLES {
        padded_audio.truncate(N_SAMPLES);
    }

    // Calculate frames we can compute from audio
    let computable_frames = (padded_audio.len() - N_FFT) / HOP_LENGTH + 1;

    // Create STFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);

    // Initialize with N_FRAMES (3000) to match Qualcomm model expectation
    let mut spectrogram = Array2::<f32>::zeros((N_FFT / 2 + 1, N_FRAMES));

    for i in 0..computable_frames.min(N_FRAMES) {
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

    // Convert to mel scale
    let mel = spectrogram_to_mel(&spectrogram)?;

    // Apply log10 (Whisper uses log10, not ln!)
    let mel_log = mel.mapv(|x| x.max(1e-10).log10());

    // Normalize (same as Whisper)
    let max_val = mel_log.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mel_normalized = mel_log.mapv(|x| (x.max(max_val - 8.0) + 4.0) / 4.0);

    Ok(mel_normalized)
}

/// Load the exact mel filterbank from Whisper (extracted from librosa)
/// Shape: [80, 201] - 80 mel bins, 201 FFT bins (N_FFT/2 + 1)
fn load_mel_filterbank() -> Array2<f32> {
    const MEL_FILTERS_DATA: &str = include_str!("mel_filters_80.txt");
    const N_FFT_BINS: usize = N_FFT / 2 + 1; // 201

    let values: Vec<f32> = MEL_FILTERS_DATA
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.trim().parse::<f32>().unwrap_or(0.0))
        .collect();

    // Reshape from flat to [80, 201]
    Array2::from_shape_vec((N_MELS, N_FFT_BINS), values).expect("Failed to reshape mel filterbank")
}

/// Convert power spectrogram to mel scale using Whisper's exact filterbank
fn spectrogram_to_mel(spec: &Array2<f32>) -> Result<Array2<f32>, Error> {
    let n_frames = spec.shape()[1];
    let filterbank = load_mel_filterbank();

    // Apply filterbank: mel = filterbank @ spec (matrix multiplication)
    // filterbank: [n_mels, n_fft_bins], spec: [n_fft_bins, n_frames]
    // result: [n_mels, n_frames]
    let mut mel = Array2::<f32>::zeros((N_MELS, n_frames));

    for i in 0..N_MELS {
        for j in 0..n_frames {
            let mut sum = 0.0;
            for k in 0..spec.shape()[0] {
                // Use power spectrum (magnitude squared)
                sum += filterbank[[i, k]] * spec[[k, j]] * spec[[k, j]];
            }
            mel[[i, j]] = sum;
        }
    }

    Ok(mel)
}

pub fn whisper_realtime(
    encoder_path: &str,
    decoder_path: &str,
    tokenizer_path: &str,
    audio_device: cpal::Device,
) -> Result<(Receiver<String>, Stream), Error> {
    let config = audio_device.default_input_config()?;
    info!("Default input config: {:?}", config);

    let input_sample_rate = config.sample_rate().0;
    let input_channels = config.channels() as usize;
    
    info!("Capturing at {} Hz, {} channels, will resample to {} Hz mono", 
          input_sample_rate, input_channels, SAMPLE_RATE);

    // Calculate buffer size for input audio (at native rate)
    let input_samples_needed = (N_SAMPLES as f64 * input_sample_rate as f64 / SAMPLE_RATE as f64) as usize;
    let input_buffer_size = input_samples_needed * input_channels;
    
    let buffer = Arc::new(Mutex::new(HeapRb::<f32>::new(input_buffer_size)));
    let buffer_producer = buffer.clone();
    let buffer_consumer = buffer.clone();

    // Use the device's native config (stereo, native sample rate)
    let stream_config = cpal::StreamConfig {
        channels: config.channels(),
        sample_rate: config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let stream = audio_device.build_input_stream(
        &stream_config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut rb = buffer_producer.lock().unwrap();
            rb.push_slice_overwrite(data);
        },
        |err| {
            warn!("Audio stream error: {}", err);
        },
        None,
    )?;

    stream.play()?;

    let (audio_chunk_sender, audio_chunk_receiver) = flume::unbounded();
    
    // Spawn task to read, convert stereo->mono, resample, and send chunks
    tokio::task::spawn(async move {
        let mut chunk = vec![0.0f32; input_buffer_size];
        
        // Create resampler (only if needed)
        let resample_ratio = SAMPLE_RATE as f64 / input_sample_rate as f64;
        let mut resampler = if input_sample_rate != SAMPLE_RATE {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            Some(SincFixedIn::<f32>::new(
                resample_ratio,
                2.0, // max relative ratio
                params,
                input_samples_needed / input_channels,
                1, // mono output
            ).expect("Failed to create resampler"))
        } else {
            None
        };

        loop {
            chunk.fill(0.0);

            {
                let mut audio_consumer = buffer_consumer.lock().unwrap();
                audio_consumer.peek_slice(&mut chunk);

                // Check energy (on raw samples)
                let energy: f64 = chunk.iter().map(|x| (*x as f64) * (*x as f64)).sum();
                
                if energy > 100.0 {
                    // Convert stereo to mono
                    let mono: Vec<f32> = if input_channels == 2 {
                        chunk.chunks(2)
                            .map(|pair| (pair[0] + pair.get(1).unwrap_or(&0.0)) / 2.0)
                            .collect()
                    } else if input_channels == 1 {
                        chunk.clone()
                    } else {
                        // Take first channel for multi-channel
                        chunk.iter().step_by(input_channels).copied().collect()
                    };

                    // Resample if needed
                    let resampled = if let Some(ref mut rs) = resampler {
                        let input = vec![mono];
                        match rs.process(&input, None) {
                            Ok(output) => output.into_iter().next().unwrap_or_default(),
                            Err(e) => {
                                warn!("Resample error: {}", e);
                                continue;
                            }
                        }
                    } else {
                        mono
                    };

                    // Pad or truncate to exactly N_SAMPLES
                    let mut final_audio = resampled;
                    if final_audio.len() < N_SAMPLES {
                        final_audio.resize(N_SAMPLES, 0.0);
                    } else if final_audio.len() > N_SAMPLES {
                        final_audio.truncate(N_SAMPLES);
                    }

                    let _ = audio_chunk_sender.send(final_audio);
                    audio_consumer.clear();
                }
            }

            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    });

    let (text_sender, text_receiver) = flume::unbounded();

    let mut model = WhisperModel::new(encoder_path, decoder_path, tokenizer_path)?;

    tokio::task::spawn_blocking(move || {
        while let Ok(chunk) = audio_chunk_receiver.recv() {
            match model.transcribe(&chunk) {
                Ok(text) => {
                    let _ = text_sender.send(text);
                    continue;
                }
                Err(e) => warn!("Error transcribing chunk: {}", e),
            };
        }
    });

    info!("Audio stream started");

    Ok((text_receiver, stream))
}
