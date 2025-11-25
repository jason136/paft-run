

# Whisper 

file download
```
https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin
```

to build on macos when getting enum or c++17 errors
```bash
CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake -B build -DCMAKE_CXX_FLAGS="-Wno-elaborated-enum-base" -DWHISPER_COREML=1 -DWHISPER_SDL2=ON
cmake --build build --config Release
```

to run the streaming example with a custom vad
```bash
git fetch origin pull/3160/head:pr-3160-vad
git checkout pr-3160-vad

cmake --build build --config Release

./build/bin/whisper-stream \
  -m ./models/ggml-base.en.bin \
  --vad \
  --vad-model ./models/ggml-silero-v5.1.2.bin \
  --vad-threshold 0.50 \
  --vad-min-speech-duration-ms 80 \
  --vad-min-silence-duration-ms 1200 \
  --vad-speech-pad-ms 150 \
  --step 500 --length 5000 --keep 300 \
  -t 8 \
  -np \
  -f /dev/stdout
```
