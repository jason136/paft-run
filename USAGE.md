# Whisper Live Speech Recognition - Usage Guide

## Overview

This implementation provides real-time speech recognition using Whisper ONNX models with the `ort` Rust bindings.

## Prerequisites

1. **ONNX Runtime Library**: Since you're using the `load-dynamic` feature, ensure ONNX Runtime is available:
   ```bash
   # macOS
   brew install onnxruntime
   
   # Or download from: https://github.com/microsoft/onnxruntime/releases
   ```

2. **Whisper ONNX Models**: Place your encoder and decoder models in the `models/` directory:
   - `models/whisper-small-encoder.onnx`
   - `models/whisper-small-decoder.onnx`

## Architecture

### Components

1. **WhisperModel** (`ort` integration)
   - Loads encoder and decoder ONNX models
   - `encode()`: Converts mel spectrogram → audio features
   - `decode()`: Autoregressive token generation
   - `transcribe()`: End-to-end audio → text pipeline

2. **Audio Capture** (using `cpal`)
   - Captures microphone input at 16kHz mono
   - Ring buffer for continuous audio streaming
   - Energy-based Voice Activity Detection (VAD)

3. **Feature Extraction**
   - STFT with Hann window (400 samples, 160 hop)
   - Mel scale conversion (80 mel bins)
   - Log scaling and normalization

### Data Flow

```
Microphone → cpal → Ring Buffer → Audio Chunks (30s)
    ↓
Energy VAD Check
    ↓
Audio Samples → STFT → Mel Spectrogram
    ↓
Encoder (ONNX) → Audio Features [1, 1500, 512]
    ↓
Decoder (ONNX) → Token IDs (autoregressive)
    ↓
Tokenizer → Text Output
```

## Key `ort` Concepts Demonstrated

### 1. Session Management

```rust
// Initialize environment once per application
ort::init()
    .with_name("whisper-live")
    .with_execution_providers([ort::CPUExecutionProvider::default().build()])
    .commit()?;

// Create optimized sessions
let encoder = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .commit_from_file(encoder_path)?;
```

**Key Points:**
- `GraphOptimizationLevel::Level3`: Maximum optimizations
- `with_intra_threads(4)`: Parallel execution within operations
- Use CPU provider by default (can switch to CUDA/CoreML)

### 2. Model Input Preparation

```rust
// Encoder: mel spectrogram [1, 80, 3000]
let mel_3d = mel.insert_axis(Axis(0));
let outputs = encoder.run(ort::inputs![mel_3d]?)?;

// Decoder: audio features + tokens
let outputs = decoder.run(ort::inputs![
    "audio_features" => audio_features.view(),
    "tokens" => tokens_array.view(),
]?)?;
```

**Key Points:**
- Use `ort::inputs![]` macro for input specification
- Named inputs vs positional inputs
- ndarray integration for tensor operations

### 3. Output Extraction

```rust
let outputs = session.run(ort::inputs![input]?)?;
let tensor = outputs[0].try_extract_tensor::<f32>()?;
let data = tensor.view(); // ndarray view
```

**Key Points:**
- `try_extract_tensor::<T>()`: Extract typed tensor
- `.view()`: Get ndarray view without copying
- Access outputs by index or name

### 4. Memory Efficiency

```rust
// Reuse session across multiple inferences
for audio_chunk in chunks {
    let result = model.transcribe(&audio_chunk)?;
}
```

**Key Points:**
- Create session once, reuse for all inferences
- Sessions are thread-safe (Arc-wrapped internally)
- Avoid recreating sessions in hot paths

## Running the Application

```bash
# Build
cargo build --release

# Run
cargo run --release
```

The application will:
1. Load the ONNX models
2. Start capturing from your default microphone
3. Process 30-second audio chunks
4. Display transcriptions when speech is detected

## Configuration

### Adjust Chunk Size

```rust
const CHUNK_LENGTH: usize = 10; // Change from 30 to 10 seconds
```

### Adjust VAD Threshold

```rust
let threshold = 0.005; // Increase for less sensitive VAD
```

### Change Model Paths

```rust
let model = WhisperModel::new(
    "path/to/encoder.onnx",
    "path/to/decoder.onnx",
)?;
```

## Getting Proper Tokenizer

The current implementation shows token IDs. For actual text:

1. **Option 1**: Use `tokenizers` crate with Whisper's vocabulary
   ```toml
   tokenizers = "0.15"
   ```

2. **Option 2**: Load Whisper's vocab.json
   ```rust
   // Download from: https://huggingface.co/openai/whisper-small/raw/main/vocab.json
   let tokenizer = Tokenizer::from_file("vocab.json")?;
   ```

3. **Option 3**: Use Python's tiktoken via PyO3

## Performance Tips

1. **Use Release Mode**: Debug builds are 10-100x slower
   ```bash
   cargo run --release
   ```

2. **Optimize Thread Count**: Match your CPU cores
   ```rust
   .with_intra_threads(num_cpus::get() as i16)?
   ```

3. **Enable Hardware Acceleration**:
   ```toml
   ort = { version = "1.16", features = ["load-dynamic", "cuda"] }
   ```
   ```rust
   .with_execution_providers([
       ort::CUDAExecutionProvider::default().build()
   ])?
   ```

4. **Batch Processing**: Process multiple chunks together if latency allows

## Troubleshooting

### "No input device available"
- Check microphone permissions in System Preferences (macOS)
- Ensure a microphone is connected

### "Failed to load encoder/decoder"
- Verify model paths are correct
- Ensure ONNX files are valid (not corrupted)
- Check ONNX Runtime is installed for `load-dynamic`

### "Model input shape mismatch"
- Verify mel spectrogram shape matches model expectations
- Check model metadata: `encoder.inputs[0].input_type`

### Poor transcription quality
- Ensure audio is 16kHz mono
- Check microphone quality
- Verify models are the correct Whisper variant
- Implement proper tokenizer (current version is placeholder)

## Advanced: Model Inspection

```rust
// Print detailed model info
println!("Encoder inputs:");
for input in &encoder.inputs {
    println!("  Name: {}", input.name);
    println!("  Type: {:?}", input.input_type);
}

println!("Encoder outputs:");
for output in &encoder.outputs {
    println!("  Name: {}", output.name);
    println!("  Type: {:?}", output.output_type);
}
```

## Next Steps

1. **Add Proper Tokenizer**: Replace `tokens_to_text()` stub
2. **Implement Streaming**: Process shorter chunks with overlap
3. **Add Language Detection**: Support multilingual models
4. **Optimize VAD**: Use Silero VAD model for better detection
5. **Add Timestamps**: Enable timestamp tokens for word-level timing
6. **GPU Acceleration**: Add CUDA/CoreML support for faster inference

## Resources

- [ort documentation](https://docs.rs/ort/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Whisper paper](https://arxiv.org/abs/2212.04356)
- [Whisper ONNX models](https://huggingface.co/models?search=whisper%20onnx)






