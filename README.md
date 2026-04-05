# CodeProject.AI-OmniVoice

A [CodeProject.AI Server](https://github.com/codeproject/CodeProject.AI-Server) plug-in that exposes [OmniVoice](https://github.com/k2-fsa/OmniVoice) – a state-of-the-art multilingual zero-shot text-to-speech model – as a REST API endpoint.

---

## Features

- **600+ languages** supported out of the box
- **Three synthesis modes** exposed via a single `/v1/sound/speak` endpoint:
  - **Voice Cloning** – clone any voice from a short reference audio clip
  - **Voice Design** – describe the desired voice with attributes (e.g. *"female, low pitch, british accent"*)
  - **Auto Voice** – let the model pick a voice automatically
- **Fine-grained controls**: adjustable speed, diffusion steps, non-verbal symbols, and pronunciation overrides
- **GPU acceleration** via CUDA (NVIDIA) or MPS (Apple Silicon); CPU fallback on unsupported hardware
- **Interactive web UI** (`explore.html`) bundled with the module for easy testing from the server dashboard

---

## Requirements

| Component | Minimum version |
|-----------|----------------|
| CodeProject.AI Server | 2.8.0 |
| Python | 3.9 |
| PyTorch | 2.0 |

A CUDA-capable GPU with ≥ 8 GB VRAM is strongly recommended for responsive inference; the module also runs on CPU or Apple Silicon MPS.

---

## Installation

Place (or symlink) this directory inside the CodeProject.AI Server modules folder and run the server's standard setup script:

```bash
# Linux / macOS
bash /path/to/CodeProject.AI-Server/src/setup.sh

# Windows
\path\to\CodeProject.AI-Server\src\setup.bat
```

The `install.sh` / `install.bat` scripts will automatically:

1. Install the required system libraries (Linux: `libsndfile1`, `ffmpeg`)
2. Install the correct PyTorch build for your platform (CUDA / MPS / CPU)
3. Install the `omnivoice` Python package (model weights are downloaded on first use from Hugging Face)

---

## API Reference

### `POST /v1/sound/speak`

Synthesize speech from text.

#### Form parameters

| Parameter  | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `text`    | string | **Yes**  | Text to synthesize |
| `ref_audio` | file | No | Reference audio file for **Voice Cloning** mode |
| `ref_text` | string | No | Transcript of `ref_audio` (auto-transcribed by Whisper if omitted) |
| `instruct` | string | No | Voice attributes for **Voice Design** mode (e.g. `"male, british accent"`) |
| `speed`   | float  | No | Speed multiplier `[0.5 – 2.0]`, default `1.0` |

#### Mode selection (automatic, evaluated in order)

1. `ref_audio` present → **Voice Cloning**
2. `instruct` present → **Voice Design**
3. Neither → **Auto Voice**

#### Response (JSON)

```json
{
  "success":     true,
  "audio":       "<base64-encoded WAV>",
  "mode":        "voice_cloning | voice_design | auto_voice",
  "inferenceMs": 1234,
  "processMs":   1250,
  "message":     "Speech synthesized successfully (voice_cloning)."
}
```

#### Example – cURL

```bash
# Auto Voice
curl -X POST http://localhost:32168/v1/sound/speak \
  -F "text=Hello, world!"

# Voice Design
curl -X POST http://localhost:32168/v1/sound/speak \
  -F "text=Hello, world!" \
  -F "instruct=female, low pitch, british accent"

# Voice Cloning
curl -X POST http://localhost:32168/v1/sound/speak \
  -F "text=Hello, world!" \
  -F "ref_audio=@speaker.wav" \
  -F "ref_text=This is the reference transcript."
```

---

## Module settings

Key environment variables (configurable via `modulesettings.json` or the server UI):

| Variable    | Default            | Description |
|------------|-------------------|-------------|
| `MODEL_NAME` | `k2-fsa/OmniVoice` | Hugging Face model ID or local path |
| `NUM_STEPS` | `32`               | Diffusion steps (16 = faster, 64 = higher quality) |
| `SPEED`     | `1.0`              | Default speech speed |
| `USE_CUDA`  | `True`             | Enable GPU inference when available |

---

## License

This plug-in is released under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).  
OmniVoice itself is also released under the Apache 2.0 License.
